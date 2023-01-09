
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

	orientation(fra2gt_0,150,96). coords(fra2gt_0,196,159,96).  visible(fra2gt_0,96). happensAt(walking(fra2gt_0),96). orientation(fra2gt_5,150,96). coords(fra2gt_5,30,121,96).  visible(fra2gt_5,96). happensAt(walking(fra2gt_5),96).
orientation(fra2gt_0,150,97). coords(fra2gt_0,197,162,97).  visible(fra2gt_0,97). happensAt(walking(fra2gt_0),97). orientation(fra2gt_5,150,97). coords(fra2gt_5,30,122,97).  visible(fra2gt_5,97). happensAt(walking(fra2gt_5),97).
orientation(fra2gt_0,150,98). coords(fra2gt_0,198,164,98).  visible(fra2gt_0,98). happensAt(walking(fra2gt_0),98). orientation(fra2gt_5,150,98). coords(fra2gt_5,30,122,98).  visible(fra2gt_5,98). happensAt(walking(fra2gt_5),98).
orientation(fra2gt_0,150,99). coords(fra2gt_0,199,166,99).  visible(fra2gt_0,99). happensAt(walking(fra2gt_0),99). orientation(fra2gt_5,150,99). coords(fra2gt_5,29,123,99).  visible(fra2gt_5,99). happensAt(walking(fra2gt_5),99).
orientation(fra2gt_0,150,100). coords(fra2gt_0,200,168,100).  visible(fra2gt_0,100). happensAt(walking(fra2gt_0),100). orientation(fra2gt_5,150,100). coords(fra2gt_5,29,123,100).  visible(fra2gt_5,100). happensAt(walking(fra2gt_5),100).
orientation(fra2gt_0,150,101). coords(fra2gt_0,202,169,101).  visible(fra2gt_0,101). happensAt(walking(fra2gt_0),101). orientation(fra2gt_5,150,101). coords(fra2gt_5,29,123,101).  visible(fra2gt_5,101). happensAt(walking(fra2gt_5),101).
orientation(fra2gt_0,150,102). coords(fra2gt_0,203,170,102).  visible(fra2gt_0,102). happensAt(walking(fra2gt_0),102). orientation(fra2gt_5,150,102). coords(fra2gt_5,29,123,102).  visible(fra2gt_5,102). happensAt(walking(fra2gt_5),102).
orientation(fra2gt_0,150,103). coords(fra2gt_0,203,171,103).  visible(fra2gt_0,103). happensAt(walking(fra2gt_0),103). orientation(fra2gt_5,150,103). coords(fra2gt_5,29,123,103).  visible(fra2gt_5,103). happensAt(walking(fra2gt_5),103).
orientation(fra2gt_0,152,104). coords(fra2gt_0,204,173,104).  visible(fra2gt_0,104). happensAt(walking(fra2gt_0),104). orientation(fra2gt_5,150,104). coords(fra2gt_5,29,123,104).  visible(fra2gt_5,104). happensAt(walking(fra2gt_5),104).
orientation(fra2gt_0,166,105). coords(fra2gt_0,205,175,105).  visible(fra2gt_0,105). happensAt(walking(fra2gt_0),105). orientation(fra2gt_5,150,105). coords(fra2gt_5,29,123,105).  visible(fra2gt_5,105). happensAt(walking(fra2gt_5),105).
orientation(fra2gt_0,165,106). coords(fra2gt_0,208,177,106).  visible(fra2gt_0,106). happensAt(walking(fra2gt_0),106). orientation(fra2gt_5,150,106). coords(fra2gt_5,29,123,106).  visible(fra2gt_5,106). happensAt(walking(fra2gt_5),106).
orientation(fra2gt_0,168,107). coords(fra2gt_0,211,179,107).  visible(fra2gt_0,107). happensAt(walking(fra2gt_0),107). orientation(fra2gt_5,150,107). coords(fra2gt_5,29,124,107).  visible(fra2gt_5,107). happensAt(walking(fra2gt_5),107).
orientation(fra2gt_0,168,108). coords(fra2gt_0,215,181,108).  visible(fra2gt_0,108). happensAt(walking(fra2gt_0),108). orientation(fra2gt_5,150,108). coords(fra2gt_5,29,124,108).  visible(fra2gt_5,108). happensAt(walking(fra2gt_5),108).
orientation(fra2gt_0,168,109). coords(fra2gt_0,219,183,109).  visible(fra2gt_0,109). happensAt(walking(fra2gt_0),109). orientation(fra2gt_5,150,109). coords(fra2gt_5,29,124,109).  visible(fra2gt_5,109). happensAt(walking(fra2gt_5),109).
orientation(fra2gt_0,168,110). coords(fra2gt_0,221,186,110).  visible(fra2gt_0,110). happensAt(walking(fra2gt_0),110). orientation(fra2gt_5,150,110). coords(fra2gt_5,27,125,110).  visible(fra2gt_5,110). happensAt(walking(fra2gt_5),110).
orientation(fra2gt_0,168,111). coords(fra2gt_0,224,189,111).  visible(fra2gt_0,111). happensAt(walking(fra2gt_0),111). orientation(fra2gt_5,150,111). coords(fra2gt_5,27,125,111).  visible(fra2gt_5,111). happensAt(walking(fra2gt_5),111).
orientation(fra2gt_0,168,112). coords(fra2gt_0,225,191,112).  visible(fra2gt_0,112). happensAt(walking(fra2gt_0),112). orientation(fra2gt_5,150,112). coords(fra2gt_5,27,125,112).  visible(fra2gt_5,112). happensAt(walking(fra2gt_5),112).
orientation(fra2gt_0,168,113). coords(fra2gt_0,227,193,113).  visible(fra2gt_0,113). happensAt(walking(fra2gt_0),113). orientation(fra2gt_5,150,113). coords(fra2gt_5,27,125,113).  visible(fra2gt_5,113). happensAt(walking(fra2gt_5),113).
orientation(fra2gt_0,168,114). coords(fra2gt_0,229,196,114).  visible(fra2gt_0,114). happensAt(walking(fra2gt_0),114). orientation(fra2gt_5,150,114). coords(fra2gt_5,27,126,114).  visible(fra2gt_5,114). happensAt(walking(fra2gt_5),114).
orientation(fra2gt_0,168,115). coords(fra2gt_0,230,200,115).  visible(fra2gt_0,115). happensAt(walking(fra2gt_0),115). orientation(fra2gt_5,150,115). coords(fra2gt_5,27,126,115).  visible(fra2gt_5,115). happensAt(walking(fra2gt_5),115).
orientation(fra2gt_0,168,116). coords(fra2gt_0,231,203,116).  visible(fra2gt_0,116). happensAt(walking(fra2gt_0),116). orientation(fra2gt_5,150,116). coords(fra2gt_5,27,126,116).  visible(fra2gt_5,116). happensAt(walking(fra2gt_5),116).
orientation(fra2gt_0,175,117). coords(fra2gt_0,232,207,117).  visible(fra2gt_0,117). happensAt(walking(fra2gt_0),117). orientation(fra2gt_5,150,117). coords(fra2gt_5,27,126,117).  visible(fra2gt_5,117). happensAt(walking(fra2gt_5),117).
orientation(fra2gt_0,9,118). coords(fra2gt_0,233,211,118).  visible(fra2gt_0,118). happensAt(walking(fra2gt_0),118). orientation(fra2gt_5,150,118). coords(fra2gt_5,27,128,118).  visible(fra2gt_5,118). happensAt(walking(fra2gt_5),118).
orientation(fra2gt_0,9,119). coords(fra2gt_0,234,215,119).  visible(fra2gt_0,119). happensAt(walking(fra2gt_0),119). orientation(fra2gt_5,150,119). coords(fra2gt_5,27,128,119).  visible(fra2gt_5,119). happensAt(walking(fra2gt_5),119).
orientation(fra2gt_0,9,120). coords(fra2gt_0,236,218,120).  visible(fra2gt_0,120). happensAt(walking(fra2gt_0),120). orientation(fra2gt_5,150,120). coords(fra2gt_5,26,129,120).  visible(fra2gt_5,120). happensAt(walking(fra2gt_5),120).
#show holdsAt/2.