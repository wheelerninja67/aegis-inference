use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" little-endian
const GGUF_ALIGNMENT: usize = 32;   // default alignment; can be overridden by metadata

#[derive(Debug)]
#[allow(dead_code)]
pub enum GgufType {
    Uint8   = 0,
    Int8    = 1,
    Uint16  = 2,
    Int16   = 3,
    Uint32  = 4,
    Int32   = 5,
    Float32 = 6,
    Bool    = 7,
    String  = 8,
    Array   = 9,
    Uint64  = 10,
    Int64   = 11,
    Float64 = 12,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name:        String,
    pub n_dims:      u32,
    pub dimensions:  Vec<u64>,   // [n_elements_dim_0, dim_1, ...]
    pub ggml_type:   u32,        // quantization type (2 = Q4_0, 14 = Q4_K, etc.)
    pub data_offset: u64,        // byte offset from start of tensor data section
}

#[derive(Debug)]
pub struct GgufHeader {
    pub version:     u32,
    pub tensor_count: u64,
    pub metadata:    std::collections::HashMap<String, MetadataValue>,
    pub tensors:     Vec<TensorInfo>,
    /// Byte offset where tensor data actually starts (after alignment padding)
    pub data_start:  u64,
}

#[derive(Debug, Clone)]
pub enum MetadataValue {
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    String(String),
    Uint64(u64),
    Bool(bool),
    Array(Vec<MetadataValue>),
    Other,
}

pub struct GgufParser {
    reader: BufReader<File>,
    pub header: GgufHeader,
}

impl GgufParser {
    pub fn open(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut reader = BufReader::with_capacity(4 * 1024 * 1024, file); // 4MB read buffer

        // --- Header ---
        let magic = read_u32_le(&mut reader)?;
        if magic != GGUF_MAGIC {
            return Err(format!("bad magic: 0x{:08X}", magic).into());
        }

        let version      = read_u32_le(&mut reader)?;
        let tensor_count = read_u64_le(&mut reader)?;
        let kv_count     = read_u64_le(&mut reader)?;

        // --- Metadata Key-Value Pairs ---
        let mut metadata = std::collections::HashMap::new();
        for _ in 0..kv_count {
            let key = read_gguf_string(&mut reader)?;
            let val_type = read_u32_le(&mut reader)?;
            let value = read_metadata_value(&mut reader, val_type)?;
            metadata.insert(key, value);
        }

        // Check for custom alignment in metadata (key: "general.alignment")
        let alignment = match metadata.get("general.alignment") {
            Some(MetadataValue::UInt32(a)) => *a as usize,
            _ => GGUF_ALIGNMENT,
        };

        // --- Tensor Info ---
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name   = read_gguf_string(&mut reader)?;
            let n_dims = read_u32_le(&mut reader)?;
            let mut dimensions = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dimensions.push(read_u64_le(&mut reader)?);
            }
            let ggml_type   = read_u32_le(&mut reader)?;
            let data_offset = read_u64_le(&mut reader)?;
            tensors.push(TensorInfo { name, n_dims, dimensions, ggml_type, data_offset });
        }

        // --- Compute data_start: align current position to `alignment` ---
        let current_pos = reader.stream_position()?;
        let data_start = (current_pos + (alignment as u64 - 1)) & !(alignment as u64 - 1);

        let header = GgufHeader { version, tensor_count, metadata, tensors, data_start };
        Ok(GgufParser { reader, header })
    }

    /// Stream a specific tensor's raw bytes into caller-provided buffer.
    /// Returns immediately if buffer too small. Zero-copy into your mmap region.
    pub unsafe fn read_tensor_into(
        &mut self,
        tensor_idx: usize,
        dest: *mut u8,
        dest_len: usize,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let info = &self.header.tensors[tensor_idx];
        let byte_size = self.tensor_byte_size(tensor_idx);

        if dest_len < byte_size {
            return Err(format!("buffer too small: need {}, have {}", byte_size, dest_len).into());
        }

        let abs_offset = self.header.data_start + info.data_offset;
        self.reader.seek(SeekFrom::Start(abs_offset))?;

        let dest_slice = unsafe { std::slice::from_raw_parts_mut(dest, byte_size) };
        self.reader.read_exact(dest_slice)?;
        Ok(byte_size)
    }

    pub fn tensor_byte_size(&self, idx: usize) -> usize {
        let info = &self.header.tensors[idx];
        let n_elements: u64 = info.dimensions.iter().product();
        match info.ggml_type {
            0  => n_elements as usize,             // F32: 4 bytes, but n_elements * 4 after
            1  => n_elements as usize * 2,         // F16
            2  => (n_elements as usize / 32) * 18, // Q4_0
             8 => (n_elements as usize / 32) * 34, // Q8_0 <- your ternary weights go here
            14 => (n_elements as usize / 256) * 144, // Q4_K
            30 => n_elements as usize * 2,         // BF16
            t  => panic!("unknown GGML type {}", t),
        }
    }
}

// --- Primitive readers (all little-endian per GGUF spec) ---
fn read_u32_le(r: &mut BufReader<File>) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}
fn read_u64_le(r: &mut BufReader<File>) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}
fn read_gguf_string(r: &mut BufReader<File>) -> std::io::Result<String> {
    let len = read_u64_le(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}
fn read_metadata_value(
    r: &mut BufReader<File>,
    val_type: u32,
) -> Result<MetadataValue, Box<dyn std::error::Error>> {
    Ok(match val_type {
        0 | 1 | 7 => { // UINT8, INT8, BOOL
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
            if val_type == 7 { MetadataValue::Bool(b[0] != 0) } else { MetadataValue::Other }
        }
        2 | 3 => { // UINT16, INT16
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
            MetadataValue::Other
        }
        4 => MetadataValue::UInt32(read_u32_le(r)?),
        5 => MetadataValue::Int32(read_u32_le(r)? as i32),
        6 => { 
            let mut b = [0u8; 4]; 
            r.read_exact(&mut b)?; 
            MetadataValue::Float32(f32::from_le_bytes(b)) 
        }
        8 => MetadataValue::String(read_gguf_string(r)?),
        9 => { // ARRAY
            let arr_type = read_u32_le(r)?;
            let arr_len = read_u64_le(r)? as usize;
            let mut arr = Vec::with_capacity(arr_len);
            for _ in 0..arr_len {
                arr.push(read_metadata_value(r, arr_type)?);
            }
            MetadataValue::Array(arr)
        }
        10 => MetadataValue::Uint64(read_u64_le(r)?),
        11 | 12 => { // INT64, FLOAT64
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            MetadataValue::Other
        }
        _ => return Err(format!("unknown metadata type: {}", val_type).into()),
    })
}
