use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;
use memmap2::{Mmap, MmapOptions};

/// Magic number for GGUF files ("GGUF" in ASCII)
const GGUF_MAGIC: u32 = 0x46554747;

pub struct GgufHeader {
    pub magic: u32,
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
}

pub struct GgufParser {
    file: File,
    pub header: GgufHeader,
    mmap: Option<Mmap>,
}

impl GgufParser {
    /// Opens and validates a GGUF file, extracting the core header metadata.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut file = File::open(path)?;
        
        let mut magic_bytes = [0u8; 4];
        file.read_exact(&mut magic_bytes)?;
        let magic = u32::from_le_bytes(magic_bytes);
        
        if magic != GGUF_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid GGUF Magic Number. Is this a corrupted model?",
            ));
        }

        let mut version_bytes = [0u8; 4];
        file.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);

        let mut tensor_count_bytes = [0u8; 8];
        file.read_exact(&mut tensor_count_bytes)?;
        let tensor_count = u64::from_le_bytes(tensor_count_bytes);

        let mut kv_count_bytes = [0u8; 8];
        file.read_exact(&mut kv_count_bytes)?;
        let kv_count = u64::from_le_bytes(kv_count_bytes);

        let header = GgufHeader {
            magic,
            version,
            tensor_count,
            kv_count,
        };

        let mut parser = Self { file, header, mmap: None };
        
        // Temporarily bypassed KV extraction because the GGUF array offsets 
        // cause pointer misalignment and 20GB string allocation panics.
        // parser.extract_kv_keys()?;

        Ok(parser)
    }

    /// Reads the length-prefixed strings from the GGUF binary format.
    fn read_gguf_string(&mut self) -> io::Result<String> {
        let mut len_bytes = [0u8; 8];
        self.file.read_exact(&mut len_bytes)?;
        let len = u64::from_le_bytes(len_bytes) as usize;

        let mut str_bytes = vec![0u8; len];
        self.file.read_exact(&mut str_bytes)?;
        
        String::from_utf8(str_bytes).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "Invalid UTF-8 in GGUF String")
        })
    }

    /// Scans the Key-Value metadata section of the GGUF file to extract keys.
    /// This allows us to identify the model architecture (e.g., Llama) and 
    /// tensor layouts before we memory-map the massive payload.
    fn extract_kv_keys(&mut self) -> io::Result<()> {
        println!("[*] Extracting Metadata keys from GGUF Header...");
        for _ in 0..self.header.kv_count {
            let key = self.read_gguf_string()?;
            
            // Read value type (u32)
            let mut type_bytes = [0u8; 4];
            self.file.read_exact(&mut type_bytes)?;
            let value_type = u32::from_le_bytes(type_bytes);
            
            // For now, we skip the actual value payload because GGUF values 
            // have 20+ different binary layouts (arrays, floats, ints, strings).
            // We just need to advance the file pointer to find the tensor offsets.
            self.skip_gguf_value(value_type)?;
            
            // println!("[-] Found Metadata Key: {}", key);
        }
        println!("[+] Successfully parsed {} KV metadata pairs.", self.header.kv_count);
        Ok(())
    }

    /// Skips over a GGUF value based on its type enum to advance the binary stream.
    fn skip_gguf_value(&mut self, value_type: u32) -> io::Result<()> {
        let skip_bytes = match value_type {
            0 => 1,  // UINT8
            1 => 1,  // INT8
            2 => 2,  // UINT16
            3 => 2,  // INT16
            4 => 4,  // UINT32
            5 => 4,  // INT32
            6 => 4,  // FLOAT32
            7 => 1,  // BOOL
            8 => {   // STRING
                let mut len_bytes = [0u8; 8];
                self.file.read_exact(&mut len_bytes)?;
                let len = u64::from_le_bytes(len_bytes) as i64;
                self.file.seek(SeekFrom::Current(len))?;
                return Ok(());
            },
            9 => 8,  // UINT64
            10 => 8, // INT64
            11 => 8, // FLOAT64
            _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "Unsupported GGUF Value Type / Arrays not yet implemented.")),
        };
        self.file.seek(SeekFrom::Current(skip_bytes))?;
        Ok(())
    }

    /// Zero-copy memory mapping (mmap) of the entire model payload.
    /// This bypasses standard OS memory allocation, allowing multi-gigabyte
    /// weight matrices to be streamed directly into the CPU's cache.
    pub fn map_tensors(&mut self) -> io::Result<()> {
        println!("[*] Initiating Zero-Copy Memory Map of GGUF Payload...");
        
        // Map the entire file as read-only memory directly from the NVMe/SSD.
        let mmap = unsafe { 
            MmapOptions::new()
                .populate() // MAP_POPULATE: Prefault the pages to eliminate cold-start latency
                .map(&self.file)? 
        };
        
        println!("[+] Successfully mapped {} bytes directly to physical memory.", mmap.len());
        self.mmap = Some(mmap);
        
        Ok(())
    }

    /// Returns a raw slice of the memory-mapped bytes for ternary quantization.
    pub fn raw_bytes(&self) -> Option<&[u8]> {
        self.mmap.as_deref()
    }
}
