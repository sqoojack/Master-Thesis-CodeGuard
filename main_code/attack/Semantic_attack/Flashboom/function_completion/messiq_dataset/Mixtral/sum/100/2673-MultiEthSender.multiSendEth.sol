 function multiSendEth(uint256 amount, address[] memory list) external returns (bool) {
 uint256 totalList = list.length;
 require(address(this).balance > amount.mul(totalList));

 for (uint256 i = 0; i < list.length; i++) {
 require(list[i]!= address(0));
 require(list[i].send(amount));

 emit Send(amount, list[i]);
 }

 return true;
}