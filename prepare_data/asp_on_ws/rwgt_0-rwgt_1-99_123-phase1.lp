
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(rwgt_0,148,99). coords(rwgt_0,235,195,99).  visible(rwgt_0,99). happensAt(walking(rwgt_0),99). orientation(rwgt_1,149,99). coords(rwgt_1,180,153,99).  visible(rwgt_1,99). happensAt(walking(rwgt_1),99).
orientation(rwgt_0,148,100). coords(rwgt_0,233,192,100).  visible(rwgt_0,100). happensAt(walking(rwgt_0),100). orientation(rwgt_1,149,100). coords(rwgt_1,178,151,100).  visible(rwgt_1,100). happensAt(walking(rwgt_1),100).
orientation(rwgt_0,150,101). coords(rwgt_0,231,189,101).  visible(rwgt_0,101). happensAt(walking(rwgt_0),101). orientation(rwgt_1,149,101). coords(rwgt_1,176,149,101).  visible(rwgt_1,101). happensAt(walking(rwgt_1),101).
orientation(rwgt_0,148,102). coords(rwgt_0,228,186,102).  visible(rwgt_0,102). happensAt(walking(rwgt_0),102). orientation(rwgt_1,149,102). coords(rwgt_1,174,147,102).  visible(rwgt_1,102). happensAt(walking(rwgt_1),102).
orientation(rwgt_0,147,103). coords(rwgt_0,226,183,103).  visible(rwgt_0,103). happensAt(walking(rwgt_0),103). orientation(rwgt_1,149,103). coords(rwgt_1,172,144,103).  visible(rwgt_1,103). happensAt(walking(rwgt_1),103).
orientation(rwgt_0,147,104). coords(rwgt_0,224,179,104).  visible(rwgt_0,104). happensAt(walking(rwgt_0),104). orientation(rwgt_1,149,104). coords(rwgt_1,171,140,104).  visible(rwgt_1,104). happensAt(walking(rwgt_1),104).
orientation(rwgt_0,147,105). coords(rwgt_0,223,177,105).  visible(rwgt_0,105). happensAt(walking(rwgt_0),105). orientation(rwgt_1,149,105). coords(rwgt_1,170,138,105).  visible(rwgt_1,105). happensAt(walking(rwgt_1),105).
orientation(rwgt_0,147,106). coords(rwgt_0,223,176,106).  visible(rwgt_0,106). happensAt(walking(rwgt_0),106). orientation(rwgt_1,149,106). coords(rwgt_1,170,135,106).  visible(rwgt_1,106). happensAt(walking(rwgt_1),106).
orientation(rwgt_0,147,107). coords(rwgt_0,222,175,107).  visible(rwgt_0,107). happensAt(walking(rwgt_0),107). orientation(rwgt_1,149,107). coords(rwgt_1,170,135,107).  visible(rwgt_1,107). happensAt(walking(rwgt_1),107).
orientation(rwgt_0,147,108). coords(rwgt_0,221,175,108).  visible(rwgt_0,108). happensAt(walking(rwgt_0),108). orientation(rwgt_1,149,108). coords(rwgt_1,169,134,108).  visible(rwgt_1,108). happensAt(walking(rwgt_1),108).
orientation(rwgt_0,147,109). coords(rwgt_0,221,173,109).  visible(rwgt_0,109). happensAt(walking(rwgt_0),109). orientation(rwgt_1,149,109). coords(rwgt_1,169,133,109).  visible(rwgt_1,109). happensAt(walking(rwgt_1),109).
orientation(rwgt_0,147,110). coords(rwgt_0,220,171,110).  visible(rwgt_0,110). happensAt(walking(rwgt_0),110). orientation(rwgt_1,149,110). coords(rwgt_1,168,133,110).  visible(rwgt_1,110). happensAt(walking(rwgt_1),110).
orientation(rwgt_0,147,111). coords(rwgt_0,219,170,111).  visible(rwgt_0,111). happensAt(walking(rwgt_0),111). orientation(rwgt_1,149,111). coords(rwgt_1,168,132,111).  visible(rwgt_1,111). happensAt(walking(rwgt_1),111).
orientation(rwgt_0,147,112). coords(rwgt_0,218,168,112).  visible(rwgt_0,112). happensAt(walking(rwgt_0),112). orientation(rwgt_1,149,112). coords(rwgt_1,168,130,112).  visible(rwgt_1,112). happensAt(walking(rwgt_1),112).
orientation(rwgt_0,147,113). coords(rwgt_0,216,167,113).  visible(rwgt_0,113). happensAt(walking(rwgt_0),113). orientation(rwgt_1,149,113). coords(rwgt_1,167,128,113).  visible(rwgt_1,113). happensAt(walking(rwgt_1),113).
orientation(rwgt_0,147,114). coords(rwgt_0,214,165,114).  visible(rwgt_0,114). happensAt(walking(rwgt_0),114). orientation(rwgt_1,149,114). coords(rwgt_1,165,126,114).  visible(rwgt_1,114). happensAt(walking(rwgt_1),114).
orientation(rwgt_0,147,115). coords(rwgt_0,212,162,115).  visible(rwgt_0,115). happensAt(walking(rwgt_0),115). orientation(rwgt_1,149,115). coords(rwgt_1,163,123,115).  visible(rwgt_1,115). happensAt(walking(rwgt_1),115).
orientation(rwgt_0,147,116). coords(rwgt_0,210,159,116).  visible(rwgt_0,116). happensAt(walking(rwgt_0),116). orientation(rwgt_1,149,116). coords(rwgt_1,162,121,116).  visible(rwgt_1,116). happensAt(walking(rwgt_1),116).
orientation(rwgt_0,147,117). coords(rwgt_0,207,157,117).  visible(rwgt_0,117). happensAt(walking(rwgt_0),117). orientation(rwgt_1,149,117). coords(rwgt_1,160,119,117).  visible(rwgt_1,117). happensAt(walking(rwgt_1),117).
orientation(rwgt_0,147,118). coords(rwgt_0,205,156,118).  visible(rwgt_0,118). happensAt(walking(rwgt_0),118). orientation(rwgt_1,149,118). coords(rwgt_1,160,117,118).  visible(rwgt_1,118). happensAt(walking(rwgt_1),118).
orientation(rwgt_0,147,119). coords(rwgt_0,205,155,119).  visible(rwgt_0,119). happensAt(walking(rwgt_0),119). orientation(rwgt_1,149,119). coords(rwgt_1,159,117,119).  visible(rwgt_1,119). happensAt(walking(rwgt_1),119).
orientation(rwgt_0,147,120). coords(rwgt_0,204,154,120).  visible(rwgt_0,120). happensAt(walking(rwgt_0),120). orientation(rwgt_1,149,120). coords(rwgt_1,158,116,120).  visible(rwgt_1,120). happensAt(walking(rwgt_1),120).
orientation(rwgt_0,147,121). coords(rwgt_0,202,154,121).  visible(rwgt_0,121). happensAt(walking(rwgt_0),121). orientation(rwgt_1,149,121). coords(rwgt_1,157,115,121).  visible(rwgt_1,121). happensAt(walking(rwgt_1),121).
orientation(rwgt_0,147,122). coords(rwgt_0,202,150,122).  visible(rwgt_0,122). happensAt(walking(rwgt_0),122). orientation(rwgt_1,147,122). coords(rwgt_1,157,114,122).  visible(rwgt_1,122). happensAt(walking(rwgt_1),122).
orientation(rwgt_0,147,123). coords(rwgt_0,202,148,123).  visible(rwgt_0,123). happensAt(walking(rwgt_0),123). orientation(rwgt_1,144,123). coords(rwgt_1,157,114,123).  visible(rwgt_1,123). happensAt(walking(rwgt_1),123).
#show holdsAt/2.