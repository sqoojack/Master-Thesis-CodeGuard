 function record_human_readable_blockhash() public one_time_use {
 bytes6[] memory word_sequence;
 word_sequence = wordbot_contract.getWords(wordcount);

 for(uint i = 0; i<wordcount; i++) {
 bytes memory word = word_sequence[i];
 bytes memory toBytes = new bytes(6);
 bytes6 word_;
 assembly {
 word_ := mload(word)
 toBytes := word_
 }
 human_readable_blockhash[i] = string(toBytes);
 }
}