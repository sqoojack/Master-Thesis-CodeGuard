 contract HelloWorld {
 function kill() public {
 selfdestruct(address(this));
 }
}