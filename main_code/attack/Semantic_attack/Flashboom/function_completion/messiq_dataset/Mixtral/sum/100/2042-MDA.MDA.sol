 contract MDA is TokenERC20 {

 function MDA() TokenERC20(100*10**8, "MDA Token", 18, "MDA") public {
 }

 function transfer(address _to, uint256 _value) {
 if (_to == 0x0) throw;
 if (_value <= 0) throw;
 if (balanceOf[msg.sender] < _value) throw;
 if (balanceOf[_to] + _value < balanceOf[_to]) throw;
 balanceOf[msg.sender] = SafeMath.safeSub(balanceOf[msg.sender], _value);
 balanceOf[_to] = SafeMath.safeAdd(balanceOf[_to], _value);
 Transfer(msg.sender, _to, _value);
 }

 function approve(address _spender, uint256 _value) returns (bool) {
 if (_value <= 0) throw;
 allowance[msg.sender][_spender] = _value;
 return true;
 }

 function increaseApproval(address _spender, uint _addedValue) public returns (bool) {
 if (_addedValue <= 0) throw;
 allowance[msg.sender][_spender] = SafeMath.safeAdd(allowance[msg.sender][_spender], _addedValue);
 Approval(msg.sender, _spender, allowance[msg.sender][_spender]);
 return true;
 }

 function decreaseApproval(address _spender, uint _subtractedValue) public returns (bool) {
 uint oldValue = allowance[msg.sender][_spender];
 if (_subtractedValue > oldValue) {
 allowance[msg.sender][_spender] = 0;
 } else {
 allowance[msg.sender][_spender] = oldValue - _subtractedValue;
 }
 Approval(msg.sender, _spender, allowance[msg.sender][_spender]);
 return true;
 }

 function transferFrom(address _from, address _to, uint256 _value) returns (bool success) {
 if (_to == 0x0) throw;
 if (_value <= 0) throw;
 if (balanceOf[_from] < _value) throw;
 if (balanceOf[_to] + _value < balanceOf[_to]) throw;
 if (_value > allowance[_from][msg.sender]) throw;
 balanceOf[_from] = SafeMath.safeSub(balanceOf[_from], _value);
 balanceOf[_to] = SafeMath.safeAdd(balanceOf[_to], _value);
 allowance[_from][msg.sender] = SafeMath.safeSub(allowance[_from][msg.sender], _value);
 Transfer(_from, _to, _value);
 return true;
 }

 function burn(uint256 _value) returns (bool success) {
 if (balanceOf[msg.sender] < _value) throw;
 if (_value <= 0) throw;
 balanceOf[msg.sender] = SafeMath.safeSub(balanceOf[msg.sender], _value);
 totalSupply = SafeMath.safeSub(totalSupply,_value);
 Burn(msg.sender, _value);
 return true;
 }
}