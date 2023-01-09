
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

	orientation(lb2gt_0,141,96). coords(lb2gt_0,193,123,96).  visible(lb2gt_0,96). happensAt(active(lb2gt_0),96). orientation(lb2gt_3,146,96). coords(lb2gt_3,11,51,96).  visible(lb2gt_3,96). happensAt(walking(lb2gt_3),96).
orientation(lb2gt_0,141,97). coords(lb2gt_0,193,123,97).  visible(lb2gt_0,97). happensAt(active(lb2gt_0),97). orientation(lb2gt_3,146,97). coords(lb2gt_3,11,51,97).  visible(lb2gt_3,97). happensAt(walking(lb2gt_3),97).
orientation(lb2gt_0,141,98). coords(lb2gt_0,194,123,98).  visible(lb2gt_0,98). happensAt(active(lb2gt_0),98). orientation(lb2gt_3,146,98). coords(lb2gt_3,11,51,98).  visible(lb2gt_3,98). happensAt(walking(lb2gt_3),98).
orientation(lb2gt_0,141,99). coords(lb2gt_0,194,123,99).  visible(lb2gt_0,99). happensAt(active(lb2gt_0),99). orientation(lb2gt_3,146,99). coords(lb2gt_3,11,51,99).  visible(lb2gt_3,99). happensAt(walking(lb2gt_3),99).
orientation(lb2gt_0,141,100). coords(lb2gt_0,194,123,100).  visible(lb2gt_0,100). happensAt(active(lb2gt_0),100). orientation(lb2gt_3,146,100). coords(lb2gt_3,11,52,100).  visible(lb2gt_3,100). happensAt(walking(lb2gt_3),100).
orientation(lb2gt_0,141,101). coords(lb2gt_0,194,123,101).  visible(lb2gt_0,101). happensAt(active(lb2gt_0),101). orientation(lb2gt_3,146,101). coords(lb2gt_3,11,52,101).  visible(lb2gt_3,101). happensAt(walking(lb2gt_3),101).
orientation(lb2gt_0,141,102). coords(lb2gt_0,194,123,102).  visible(lb2gt_0,102). happensAt(active(lb2gt_0),102). orientation(lb2gt_3,146,102). coords(lb2gt_3,11,52,102).  visible(lb2gt_3,102). happensAt(walking(lb2gt_3),102).
orientation(lb2gt_0,141,103). coords(lb2gt_0,194,123,103).  visible(lb2gt_0,103). happensAt(active(lb2gt_0),103). orientation(lb2gt_3,146,103). coords(lb2gt_3,10,52,103).  visible(lb2gt_3,103). happensAt(walking(lb2gt_3),103).
orientation(lb2gt_0,141,104). coords(lb2gt_0,194,123,104).  visible(lb2gt_0,104). happensAt(active(lb2gt_0),104). orientation(lb2gt_3,146,104). coords(lb2gt_3,10,52,104).  visible(lb2gt_3,104). happensAt(walking(lb2gt_3),104).
orientation(lb2gt_0,141,105). coords(lb2gt_0,194,123,105).  visible(lb2gt_0,105). happensAt(active(lb2gt_0),105). orientation(lb2gt_3,146,105). coords(lb2gt_3,10,52,105).  visible(lb2gt_3,105). happensAt(walking(lb2gt_3),105).
orientation(lb2gt_0,141,106). coords(lb2gt_0,194,123,106).  visible(lb2gt_0,106). happensAt(active(lb2gt_0),106). orientation(lb2gt_3,146,106). coords(lb2gt_3,10,52,106).  visible(lb2gt_3,106). happensAt(walking(lb2gt_3),106).
orientation(lb2gt_0,141,107). coords(lb2gt_0,194,123,107).  visible(lb2gt_0,107). happensAt(active(lb2gt_0),107). orientation(lb2gt_3,146,107). coords(lb2gt_3,10,52,107).  visible(lb2gt_3,107). happensAt(walking(lb2gt_3),107).
orientation(lb2gt_0,141,108). coords(lb2gt_0,194,123,108).  visible(lb2gt_0,108). happensAt(active(lb2gt_0),108). orientation(lb2gt_3,146,108). coords(lb2gt_3,10,53,108).  visible(lb2gt_3,108). happensAt(walking(lb2gt_3),108).
orientation(lb2gt_0,141,109). coords(lb2gt_0,194,123,109).  visible(lb2gt_0,109). happensAt(active(lb2gt_0),109). orientation(lb2gt_3,146,109). coords(lb2gt_3,10,53,109).  visible(lb2gt_3,109). happensAt(walking(lb2gt_3),109).
orientation(lb2gt_0,141,110). coords(lb2gt_0,194,123,110).  visible(lb2gt_0,110). happensAt(active(lb2gt_0),110). orientation(lb2gt_3,146,110). coords(lb2gt_3,10,54,110).  visible(lb2gt_3,110). happensAt(walking(lb2gt_3),110).
orientation(lb2gt_0,141,111). coords(lb2gt_0,194,123,111).  visible(lb2gt_0,111). happensAt(active(lb2gt_0),111). orientation(lb2gt_3,146,111). coords(lb2gt_3,10,54,111).  visible(lb2gt_3,111). happensAt(walking(lb2gt_3),111).
orientation(lb2gt_0,141,112). coords(lb2gt_0,194,123,112).  visible(lb2gt_0,112). happensAt(active(lb2gt_0),112). orientation(lb2gt_3,146,112). coords(lb2gt_3,10,54,112).  visible(lb2gt_3,112). happensAt(walking(lb2gt_3),112).
orientation(lb2gt_0,141,113). coords(lb2gt_0,194,123,113).  visible(lb2gt_0,113). happensAt(active(lb2gt_0),113). orientation(lb2gt_3,146,113). coords(lb2gt_3,10,54,113).  visible(lb2gt_3,113). happensAt(walking(lb2gt_3),113).
orientation(lb2gt_0,141,114). coords(lb2gt_0,194,123,114).  visible(lb2gt_0,114). happensAt(active(lb2gt_0),114). orientation(lb2gt_3,146,114). coords(lb2gt_3,10,54,114).  visible(lb2gt_3,114). happensAt(walking(lb2gt_3),114).
orientation(lb2gt_0,141,115). coords(lb2gt_0,194,123,115).  visible(lb2gt_0,115). happensAt(active(lb2gt_0),115). orientation(lb2gt_3,146,115). coords(lb2gt_3,10,54,115).  visible(lb2gt_3,115). happensAt(walking(lb2gt_3),115).
orientation(lb2gt_0,141,116). coords(lb2gt_0,194,123,116).  visible(lb2gt_0,116). happensAt(active(lb2gt_0),116). orientation(lb2gt_3,146,116). coords(lb2gt_3,9,55,116).  visible(lb2gt_3,116). happensAt(walking(lb2gt_3),116).
orientation(lb2gt_0,141,117). coords(lb2gt_0,194,123,117).  visible(lb2gt_0,117). happensAt(active(lb2gt_0),117). orientation(lb2gt_3,146,117). coords(lb2gt_3,9,55,117).  visible(lb2gt_3,117). happensAt(walking(lb2gt_3),117).
orientation(lb2gt_0,141,118). coords(lb2gt_0,194,123,118).  visible(lb2gt_0,118). happensAt(active(lb2gt_0),118). orientation(lb2gt_3,146,118). coords(lb2gt_3,9,56,118).  visible(lb2gt_3,118). happensAt(walking(lb2gt_3),118).
orientation(lb2gt_0,141,119). coords(lb2gt_0,194,123,119).  visible(lb2gt_0,119). happensAt(active(lb2gt_0),119). orientation(lb2gt_3,146,119). coords(lb2gt_3,9,56,119).  visible(lb2gt_3,119). happensAt(walking(lb2gt_3),119).
orientation(lb2gt_0,141,120). coords(lb2gt_0,194,123,120).  visible(lb2gt_0,120). happensAt(active(lb2gt_0),120). orientation(lb2gt_3,146,120). coords(lb2gt_3,9,56,120).  visible(lb2gt_3,120). happensAt(walking(lb2gt_3),120).
#show holdsAt/2.