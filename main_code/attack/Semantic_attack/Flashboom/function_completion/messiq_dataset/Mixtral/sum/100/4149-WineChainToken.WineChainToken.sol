 contract WineChainToken is StandardToken {
    using SafeMath for uint256;
    string public name="WineChain";
    string public symbol="WID";
    uint8 public decimals = 8;
    uint256 public totalSupply;

    constructor() public {
        totalSupply = 900000000 * 10 ** uint256(decimals);
        balances[msg.sender] = totalSupply;
    }

    function transfer(address _to, uint256 _value) public returns (bool) {
        require(_to!= address(0));
        require(_value <= balances[msg.sender]);

        balances[msg.sender] = balances[msg.sender].sub(_value);
        balances[_to] = balances[_to].add(_value);
        Transfer(msg.sender, _to, _value);
        return true;
    }

    function balanceOf(address _owner) public view returns (uint256 balance) {
        return balances[_owner];
    }

    function _transfer(address _from, address _to, uint256 _value) internal {
        require(_to!= address(0));
        require(_value <= balances[_from]);

        balances[_from] = balances[_from].sub(_value);
        balances[_to] = balances[_to].add(_value);
        Transfer(_from, _to, _value);
    }

    function _mint(address _to, uint256 _amount) internal {
        require(_to!= address(0));

        balances[_to] = balances[_to].add(_amount);
        totalSupply = totalSupply.add(_amount);
    }

    function _burn(address _from, uint256 _amount) internal {
        require(_amount <= balances[_from]);

        balances[_from] = balances[_from].sub(_amount);
        totalSupply = totalSupply.sub(_amount);
    }
}