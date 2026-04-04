 contract UCToken is TokenERC20 {

 function UCToken(uint256 initialSupply, string memory tokenName, string memory tokenSymbol) public {}

}

function _transfer(address _from, address _to, uint _value) internal {
 require(_to!= 0x0,);
 require(balanceOf[_from] >= _value,);
 require(balanceOf[_to] + _value > balanceOf[_to],);
 uint previousBalances = balanceOf[_from] + balanceOf[_to];
 balanceOf[_from] -= _value;
 balanceOf[_to] += _value;
 Transfer(_from, _to, _value);
 assert(balanceOf[_from] + balanceOf[_to] == previousBalances,);
}

function transfer(address _to, uint256 _value) public {
 _transfer(msg.sender, _to, _value);
}

function transferFrom(address _from, address _to, uint256 _value) public returns (bool) {
 require(_value <= allowance[_from][msg.sender],);
 allowance[_from][msg.sender] -= _value;
 _transfer(_from, _to, _value);
 return true;
}

function approve(address _spender, uint256 _value) public returns (bool) {
 allowance[msg.sender][_spender] = _value;
 return true;
}