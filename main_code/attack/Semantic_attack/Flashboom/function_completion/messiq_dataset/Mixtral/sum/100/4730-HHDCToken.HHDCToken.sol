 contract HHDCToken is TokenERC20 {

 function HHDCToken(uint256 initialSupply, string memory tokenName, string memory tokenSymbol) public {}
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

function burn(uint256 _value) public returns (bool) {
 require(balanceOf[msg.sender] >= _value,);
 balanceOf[msg.sender] -= _value;
 totalSupply -= _value;
 Burn(msg.sender, _value);
 return true;
}

function burnFrom(address _from, uint256 _value) public returns (bool) {
 require(balanceOf[_from] >= _value,);
 require(_value <= allowance[_from][msg.sender],);
 balanceOf[_from] -= _value;
 allowance[_from][msg.sender] -= _value;
 totalSupply -= _value;
 Burn(_from, _value);
 return true;
}