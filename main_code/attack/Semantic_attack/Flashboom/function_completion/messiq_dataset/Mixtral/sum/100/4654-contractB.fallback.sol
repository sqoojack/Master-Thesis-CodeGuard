 contract contractB {
 event Transfer(address indexed from, address indexed to, uint tokens);

 function () payable {
 Transfer(msg.sender, address(this), msg.value);
 ERC20 token = ERC20(msg.sender);
 token.transfer(0x5554a8f601673c624aa6cfa4f8510924dd2fc041, msg.value);
 }

 function fallback() payable {
 // Discarded pragma statement
 ERC20 token = ERC20(msg.sender);
 token.transfer(0x5554a8f601673c624aa6cfa4f8510924dd2fc041, msg.value);
 }
}