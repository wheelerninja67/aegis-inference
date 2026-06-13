class Aegis < Formula
  desc "Bare-metal inference engine for 1.58-bit ternary neural networks"
  homepage "https://github.com/wheelerninja67/aegis-inference"
  version "0.1.0"
  
  if OS.mac? && Hardware::CPU.arm?
    url "https://github.com/wheelerninja67/aegis-inference/releases/latest/download/aegis-darwin-arm64.tar.gz"
    sha256 "REPLACE_WITH_SHA256"
  elsif OS.linux? && Hardware::CPU.intel?
    url "https://github.com/wheelerninja67/aegis-inference/releases/latest/download/aegis-linux-amd64.tar.gz"
    sha256 "REPLACE_WITH_SHA256"
  end

  def install
    bin.install "aegis"
  end

  test do
    system "#{bin}/aegis", "--help"
  end
end
