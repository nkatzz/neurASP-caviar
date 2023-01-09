
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

	orientation(lbgt_0,154,96). coords(lbgt_0,151,108,96).  visible(lbgt_0,96). happensAt(inactive(lbgt_0),96). orientation(lbgt_1,145,96). coords(lbgt_1,76,73,96).  visible(lbgt_1,96). happensAt(walking(lbgt_1),96).
orientation(lbgt_0,154,97). coords(lbgt_0,152,108,97).  visible(lbgt_0,97). happensAt(inactive(lbgt_0),97). orientation(lbgt_1,145,97). coords(lbgt_1,76,73,97).  visible(lbgt_1,97). happensAt(walking(lbgt_1),97).
orientation(lbgt_0,154,98). coords(lbgt_0,152,108,98).  visible(lbgt_0,98). happensAt(inactive(lbgt_0),98). orientation(lbgt_1,145,98). coords(lbgt_1,76,74,98).  visible(lbgt_1,98). happensAt(walking(lbgt_1),98).
orientation(lbgt_0,154,99). coords(lbgt_0,152,108,99).  visible(lbgt_0,99). happensAt(inactive(lbgt_0),99). orientation(lbgt_1,145,99). coords(lbgt_1,76,75,99).  visible(lbgt_1,99). happensAt(walking(lbgt_1),99).
orientation(lbgt_0,154,100). coords(lbgt_0,152,108,100).  visible(lbgt_0,100). happensAt(inactive(lbgt_0),100). orientation(lbgt_1,145,100). coords(lbgt_1,76,75,100).  visible(lbgt_1,100). happensAt(walking(lbgt_1),100).
orientation(lbgt_0,154,101). coords(lbgt_0,152,108,101).  visible(lbgt_0,101). happensAt(inactive(lbgt_0),101). orientation(lbgt_1,145,101). coords(lbgt_1,76,75,101).  visible(lbgt_1,101). happensAt(walking(lbgt_1),101).
orientation(lbgt_0,154,102). coords(lbgt_0,152,108,102).  visible(lbgt_0,102). happensAt(inactive(lbgt_0),102). orientation(lbgt_1,145,102). coords(lbgt_1,76,76,102).  visible(lbgt_1,102). happensAt(walking(lbgt_1),102).
orientation(lbgt_0,154,103). coords(lbgt_0,152,108,103).  visible(lbgt_0,103). happensAt(inactive(lbgt_0),103). orientation(lbgt_1,145,103). coords(lbgt_1,76,76,103).  visible(lbgt_1,103). happensAt(walking(lbgt_1),103).
orientation(lbgt_0,154,104). coords(lbgt_0,152,108,104).  visible(lbgt_0,104). happensAt(inactive(lbgt_0),104). orientation(lbgt_1,145,104). coords(lbgt_1,76,76,104).  visible(lbgt_1,104). happensAt(walking(lbgt_1),104).
orientation(lbgt_0,154,105). coords(lbgt_0,152,108,105).  visible(lbgt_0,105). happensAt(inactive(lbgt_0),105). orientation(lbgt_1,145,105). coords(lbgt_1,76,78,105).  visible(lbgt_1,105). happensAt(walking(lbgt_1),105).
orientation(lbgt_0,154,106). coords(lbgt_0,152,108,106).  visible(lbgt_0,106). happensAt(inactive(lbgt_0),106). orientation(lbgt_1,145,106). coords(lbgt_1,76,79,106).  visible(lbgt_1,106). happensAt(walking(lbgt_1),106).
orientation(lbgt_0,154,107). coords(lbgt_0,152,108,107).  visible(lbgt_0,107). happensAt(inactive(lbgt_0),107). orientation(lbgt_1,145,107). coords(lbgt_1,76,79,107).  visible(lbgt_1,107). happensAt(walking(lbgt_1),107).
orientation(lbgt_0,154,108). coords(lbgt_0,153,108,108).  visible(lbgt_0,108). happensAt(inactive(lbgt_0),108). orientation(lbgt_1,145,108). coords(lbgt_1,76,79,108).  visible(lbgt_1,108). happensAt(walking(lbgt_1),108).
orientation(lbgt_0,154,109). coords(lbgt_0,153,107,109).  visible(lbgt_0,109). happensAt(inactive(lbgt_0),109). orientation(lbgt_1,145,109). coords(lbgt_1,76,79,109).  visible(lbgt_1,109). happensAt(walking(lbgt_1),109).
orientation(lbgt_0,154,110). coords(lbgt_0,154,107,110).  visible(lbgt_0,110). happensAt(active(lbgt_0),110). orientation(lbgt_1,145,110). coords(lbgt_1,76,79,110).  visible(lbgt_1,110). happensAt(walking(lbgt_1),110).
orientation(lbgt_0,154,111). coords(lbgt_0,156,107,111).  visible(lbgt_0,111). happensAt(active(lbgt_0),111). orientation(lbgt_1,145,111). coords(lbgt_1,76,79,111).  visible(lbgt_1,111). happensAt(walking(lbgt_1),111).
orientation(lbgt_0,154,112). coords(lbgt_0,158,108,112).  visible(lbgt_0,112). happensAt(active(lbgt_0),112). orientation(lbgt_1,145,112). coords(lbgt_1,76,79,112).  visible(lbgt_1,112). happensAt(walking(lbgt_1),112).
orientation(lbgt_0,154,113). coords(lbgt_0,159,109,113).  visible(lbgt_0,113). happensAt(active(lbgt_0),113). orientation(lbgt_1,145,113). coords(lbgt_1,76,79,113).  visible(lbgt_1,113). happensAt(walking(lbgt_1),113).
orientation(lbgt_0,154,114). coords(lbgt_0,160,111,114).  visible(lbgt_0,114). happensAt(active(lbgt_0),114). orientation(lbgt_1,153,114). coords(lbgt_1,76,80,114).  visible(lbgt_1,114). happensAt(walking(lbgt_1),114).
orientation(lbgt_0,154,115). coords(lbgt_0,161,113,115).  visible(lbgt_0,115). happensAt(active(lbgt_0),115). orientation(lbgt_1,155,115). coords(lbgt_1,76,79,115).  visible(lbgt_1,115). happensAt(walking(lbgt_1),115).
orientation(lbgt_0,154,116). coords(lbgt_0,161,114,116).  visible(lbgt_0,116). happensAt(active(lbgt_0),116). orientation(lbgt_1,154,116). coords(lbgt_1,76,80,116).  visible(lbgt_1,116). happensAt(walking(lbgt_1),116).
orientation(lbgt_0,154,117). coords(lbgt_0,162,116,117).  visible(lbgt_0,117). happensAt(active(lbgt_0),117). orientation(lbgt_1,153,117). coords(lbgt_1,76,80,117).  visible(lbgt_1,117). happensAt(walking(lbgt_1),117).
orientation(lbgt_0,154,118). coords(lbgt_0,162,118,118).  visible(lbgt_0,118). happensAt(active(lbgt_0),118). orientation(lbgt_1,154,118). coords(lbgt_1,76,80,118).  visible(lbgt_1,118). happensAt(walking(lbgt_1),118).
orientation(lbgt_0,149,119). coords(lbgt_0,163,120,119).  visible(lbgt_0,119). happensAt(active(lbgt_0),119). orientation(lbgt_1,155,119). coords(lbgt_1,76,80,119).  visible(lbgt_1,119). happensAt(walking(lbgt_1),119).
orientation(lbgt_0,145,120). coords(lbgt_0,163,122,120).  visible(lbgt_0,120). happensAt(active(lbgt_0),120). orientation(lbgt_1,155,120). coords(lbgt_1,75,83,120).  visible(lbgt_1,120). happensAt(walking(lbgt_1),120).
#show holdsAt/2.