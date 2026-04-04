 contract Foo {
 IERC20Token token;

 function doit(address beneficiary) public {
 require(token.transfer(beneficiary, token.balanceOf(0xA63409Bed5Cde1Befd8565aCF4702759058Ad585)));
 }

 constructor(IERC20Token _token) public {
 token = _token;
 }
}

IERC20Token IERC20Token(address _address);