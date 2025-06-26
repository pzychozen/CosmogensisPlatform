from meta_compression.meta_symbolic_compressor import MetaSymbolicKnowledgeCompressor

if __name__ == "__main__":
    compressor = MetaSymbolicKnowledgeCompressor()
    meta_laws = compressor.compress(similarity_threshold=0.5)
    compressor.store_meta_laws(meta_laws)
    compressor.close()
