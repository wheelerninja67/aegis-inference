use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;
use memmap2::{Mmap, MmapOptions};

const GGUF_MAGIC: u32 = 0x46554747;

#[derive(Clone, Debug)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    Array(Vec<MetadataValue>),
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
}

pub struct GgufHeader {
    pub magic: u32,
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: Vec<TensorInfo>,
    pub alignment: u64,
}

pub struct GgufParser {
    file: File,
    pub header: GgufHeader,
    mmap: Option<Mmap>,
    tensor_data_offset: u64,
}

impl GgufParser {
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut file = File::open(path)?;
        
        let mut magic_bytes = [0u8; 4];
        file.read_exact(&mut magic_bytes)?;
        let magic = u32::from_le_bytes(magic_bytes);
        if magic != GGUF_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid GGUF Magic Number"));
        }

        let mut version_bytes = [0u8; 4];
        file.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);

        let mut tcb = [0u8; 8];
        file.read_exact(&mut tcb)?;
        let tensor_count = u64::from_le_bytes(tcb);

        let mut kvb = [0u8; 8];
        file.read_exact(&mut kvb)?;
        let kv_count = u64::from_le_bytes(kvb);

        let mut header = GgufHeader {
            magic, version, tensor_count, kv_count,
            metadata: HashMap::new(),
            tensors: Vec::new(),
            alignment: 32,
        };

        let mut parser = Self { file, header, mmap: None, tensor_data_offset: 0 };
        
        parser.extract_kv_keys()?;
        parser.extract_tensors()?;

        // Calculate tensor data offset (align up)
        let current_pos = parser.file.seek(SeekFrom::Current(0))?;
        let alignment = parser.header.alignment;
        parser.tensor_data_offset = (current_pos + alignment - 1) / alignment * alignment;

        Ok(parser)
    }

    fn read_string(&mut self) -> io::Result<String> {
        let mut len_bytes = [0u8; 8];
        self.file.read_exact(&mut len_bytes)?;
        let len = u64::from_le_bytes(len_bytes) as usize;
        let mut str_bytes = vec![0u8; len];
        self.file.read_exact(&mut str_bytes)?;
        String::from_utf8(str_bytes).map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid UTF-8"))
    }

    fn read_value(&mut self, value_type: u32) -> io::Result<MetadataValue> {
        match value_type {
            0 => { let mut b = [0u8; 1]; self.file.read_exact(&mut b)?; Ok(MetadataValue::Uint8(b[0])) },
            1 => { let mut b = [0u8; 1]; self.file.read_exact(&mut b)?; Ok(MetadataValue::Int8(b[0] as i8)) },
            2 => { let mut b = [0u8; 2]; self.file.read_exact(&mut b)?; Ok(MetadataValue::Uint16(u16::from_le_bytes(b))) },
            3 => { let mut b = [0u8; 2]; self.file.read_exact(&mut b)?; Ok(MetadataValue::Int16(i16::from_le_bytes(b))) },
            4 => { let mut b = [0u8; 4]; self.file.read_exact(&mut b)?; Ok(MetadataValue::UInt32(u32::from_le_bytes(b))) },
            5 => { let mut b = [0u8; 4]; self.file.read_exact(&mut b)?; Ok(MetadataValue::Int32(i32::from_le_bytes(b))) },
            6 => { let mut b = [0u8; 4]; self.file.read_exact(&mut b)?; Ok(MetadataValue::Float32(f32::from_le_bytes(b))) },
            7 => { let mut b = [0u8; 1]; self.file.read_exact(&mut b)?; Ok(MetadataValue::Bool(b[0] != 0)) },
            8 => { Ok(MetadataValue::String(self.read_string()?)) },
            9 => { let mut b = [0u8; 8]; self.file.read_exact(&mut b)?; Ok(MetadataValue::Uint64(u64::from_le_bytes(b))) },
            10 => { let mut b = [0u8; 8]; self.file.read_exact(&mut b)?; Ok(MetadataValue::Int64(i64::from_le_bytes(b))) },
            11 => { let mut b = [0u8; 8]; self.file.read_exact(&mut b)?; Ok(MetadataValue::Float64(f64::from_le_bytes(b))) },
            12 => {
                let mut ty_bytes = [0u8; 4];
                self.file.read_exact(&mut ty_bytes)?;
                let item_type = u32::from_le_bytes(ty_bytes);
                
                let mut len_bytes = [0u8; 8];
                self.file.read_exact(&mut len_bytes)?;
                let len = u64::from_le_bytes(len_bytes) as usize;
                
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(self.read_value(item_type)?);
                }
                Ok(MetadataValue::Array(arr))
            },
            _ => Err(io::Error::new(io::ErrorKind::InvalidData, "Unknown GGUF Value Type")),
        }
    }

    fn extract_kv_keys(&mut self) -> io::Result<()> {
        for _ in 0..self.header.kv_count {
            let key = self.read_string()?;
            let mut type_bytes = [0u8; 4];
            self.file.read_exact(&mut type_bytes)?;
            let value_type = u32::from_le_bytes(type_bytes);
            
            let val = self.read_value(value_type)?;
            if key == "general.alignment" {
                if let MetadataValue::UInt32(a) = val { self.header.alignment = a as u64; }
            }
            self.header.metadata.insert(key, val);
        }
        Ok(())
    }

    fn extract_tensors(&mut self) -> io::Result<()> {
        for _ in 0..self.header.tensor_count {
            let name = self.read_string()?;
            
            let mut n_dims_bytes = [0u8; 4];
            self.file.read_exact(&mut n_dims_bytes)?;
            let n_dims = u32::from_le_bytes(n_dims_bytes) as usize;
            
            let mut dimensions = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                let mut d = [0u8; 8];
                self.file.read_exact(&mut d)?;
                dimensions.push(u64::from_le_bytes(d));
            }
            
            let mut ty_bytes = [0u8; 4];
            self.file.read_exact(&mut ty_bytes)?;
            let ggml_type = u32::from_le_bytes(ty_bytes);
            
            let mut offset_bytes = [0u8; 8];
            self.file.read_exact(&mut offset_bytes)?;
            let offset = u64::from_le_bytes(offset_bytes);
            
            self.header.tensors.push(TensorInfo { name, dimensions, ggml_type, offset });
        }
        Ok(())
    }

    pub fn map_tensors(&mut self) -> io::Result<()> {
        let mmap = unsafe { MmapOptions::new().populate().map(&self.file)? };
        self.mmap = Some(mmap);
        Ok(())
    }

    pub fn raw_bytes(&self) -> Option<&[u8]> {
        self.mmap.as_deref()
    }

    pub fn tensor_byte_size(&self, idx: usize) -> usize {
        let t = &self.header.tensors[idx];
        let elements: u64 = t.dimensions.iter().product();
        match t.ggml_type {
            0 => (elements * 4) as usize, // F32
            8 => {
                let blocks = (elements + 31) / 32;
                (blocks * 34) as usize // Q8_0
            },
            _ => panic!("Unsupported tensor type for sizing"),
        }
    }

    pub fn read_tensor_into(&mut self, idx: usize, ptr: *mut u8, size: usize) -> io::Result<()> {
        let offset = self.tensor_data_offset + self.header.tensors[idx].offset;
        self.file.seek(SeekFrom::Start(offset))?;
        unsafe {
            let slice = std::slice::from_raw_parts_mut(ptr, size);
            self.file.read_exact(slice)?;
        }
        Ok(())
    }
}
