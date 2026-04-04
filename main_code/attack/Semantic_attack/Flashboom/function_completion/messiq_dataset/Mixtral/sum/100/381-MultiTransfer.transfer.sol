 contract MultiTransfer is Ownable {

 function transfer(address token, address[] memory to, uint256[] memory value) public onlyOwner {
 require(to.length == value.length);
 require(token!= address(0));

 ERC20 memory t = ERC20(token);
 for (uint i = 0; i < to.length; i++) {
 t.transfer(to[i], value[i]);
 }
 }
}