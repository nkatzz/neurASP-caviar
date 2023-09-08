
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

	orientation(wk2gt_2,141,94). coords(wk2gt_2,98,62,94).  visible(wk2gt_2,94). happensAt(inactive(wk2gt_2),94). orientation(wk2gt_3,6,94). coords(wk2gt_3,219,252,94).  visible(wk2gt_3,94). happensAt(walking(wk2gt_3),94).
orientation(wk2gt_2,141,95). coords(wk2gt_2,98,62,95).  visible(wk2gt_2,95). happensAt(inactive(wk2gt_2),95). orientation(wk2gt_3,6,95). coords(wk2gt_3,218,250,95).  visible(wk2gt_3,95). happensAt(walking(wk2gt_3),95).
orientation(wk2gt_2,141,96). coords(wk2gt_2,98,62,96).  visible(wk2gt_2,96). happensAt(inactive(wk2gt_2),96). orientation(wk2gt_3,6,96). coords(wk2gt_3,217,247,96).  visible(wk2gt_3,96). happensAt(walking(wk2gt_3),96).
orientation(wk2gt_2,141,97). coords(wk2gt_2,98,62,97).  visible(wk2gt_2,97). happensAt(inactive(wk2gt_2),97). orientation(wk2gt_3,6,97). coords(wk2gt_3,217,244,97).  visible(wk2gt_3,97). happensAt(walking(wk2gt_3),97).
orientation(wk2gt_2,141,98). coords(wk2gt_2,98,62,98).  visible(wk2gt_2,98). happensAt(inactive(wk2gt_2),98). orientation(wk2gt_3,176,98). coords(wk2gt_3,216,241,98).  visible(wk2gt_3,98). happensAt(walking(wk2gt_3),98).
orientation(wk2gt_2,141,99). coords(wk2gt_2,98,62,99).  visible(wk2gt_2,99). happensAt(inactive(wk2gt_2),99). orientation(wk2gt_3,176,99). coords(wk2gt_3,215,239,99).  visible(wk2gt_3,99). happensAt(walking(wk2gt_3),99).
orientation(wk2gt_2,141,100). coords(wk2gt_2,98,62,100).  visible(wk2gt_2,100). happensAt(inactive(wk2gt_2),100). orientation(wk2gt_3,167,100). coords(wk2gt_3,214,236,100).  visible(wk2gt_3,100). happensAt(walking(wk2gt_3),100).
orientation(wk2gt_2,141,101). coords(wk2gt_2,98,62,101).  visible(wk2gt_2,101). happensAt(inactive(wk2gt_2),101). orientation(wk2gt_3,167,101). coords(wk2gt_3,213,234,101).  visible(wk2gt_3,101). happensAt(walking(wk2gt_3),101).
orientation(wk2gt_2,141,102). coords(wk2gt_2,98,62,102).  visible(wk2gt_2,102). happensAt(inactive(wk2gt_2),102). orientation(wk2gt_3,167,102). coords(wk2gt_3,213,233,102).  visible(wk2gt_3,102). happensAt(walking(wk2gt_3),102).
orientation(wk2gt_2,141,103). coords(wk2gt_2,98,62,103).  visible(wk2gt_2,103). happensAt(inactive(wk2gt_2),103). orientation(wk2gt_3,167,103). coords(wk2gt_3,211,232,103).  visible(wk2gt_3,103). happensAt(walking(wk2gt_3),103).
orientation(wk2gt_2,141,104). coords(wk2gt_2,98,62,104).  visible(wk2gt_2,104). happensAt(inactive(wk2gt_2),104). orientation(wk2gt_3,167,104). coords(wk2gt_3,210,231,104).  visible(wk2gt_3,104). happensAt(walking(wk2gt_3),104).
orientation(wk2gt_2,141,105). coords(wk2gt_2,98,62,105).  visible(wk2gt_2,105). happensAt(inactive(wk2gt_2),105). orientation(wk2gt_3,167,105). coords(wk2gt_3,209,228,105).  visible(wk2gt_3,105). happensAt(walking(wk2gt_3),105).
orientation(wk2gt_2,141,106). coords(wk2gt_2,98,62,106).  visible(wk2gt_2,106). happensAt(inactive(wk2gt_2),106). orientation(wk2gt_3,167,106). coords(wk2gt_3,207,226,106).  visible(wk2gt_3,106). happensAt(walking(wk2gt_3),106).
orientation(wk2gt_2,141,107). coords(wk2gt_2,98,62,107).  visible(wk2gt_2,107). happensAt(inactive(wk2gt_2),107). orientation(wk2gt_3,167,107). coords(wk2gt_3,205,222,107).  visible(wk2gt_3,107). happensAt(walking(wk2gt_3),107).
orientation(wk2gt_2,141,108). coords(wk2gt_2,98,62,108).  visible(wk2gt_2,108). happensAt(inactive(wk2gt_2),108). orientation(wk2gt_3,167,108). coords(wk2gt_3,203,218,108).  visible(wk2gt_3,108). happensAt(walking(wk2gt_3),108).
orientation(wk2gt_2,141,109). coords(wk2gt_2,98,62,109).  visible(wk2gt_2,109). happensAt(inactive(wk2gt_2),109). orientation(wk2gt_3,168,109). coords(wk2gt_3,201,212,109).  visible(wk2gt_3,109). happensAt(walking(wk2gt_3),109).
orientation(wk2gt_2,141,110). coords(wk2gt_2,98,62,110).  visible(wk2gt_2,110). happensAt(inactive(wk2gt_2),110). orientation(wk2gt_3,168,110). coords(wk2gt_3,200,210,110).  visible(wk2gt_3,110). happensAt(walking(wk2gt_3),110).
orientation(wk2gt_2,141,111). coords(wk2gt_2,98,62,111).  visible(wk2gt_2,111). happensAt(inactive(wk2gt_2),111). orientation(wk2gt_3,168,111). coords(wk2gt_3,200,207,111).  visible(wk2gt_3,111). happensAt(walking(wk2gt_3),111).
orientation(wk2gt_2,141,112). coords(wk2gt_2,98,62,112).  visible(wk2gt_2,112). happensAt(inactive(wk2gt_2),112). orientation(wk2gt_3,168,112). coords(wk2gt_3,200,205,112).  visible(wk2gt_3,112). happensAt(walking(wk2gt_3),112).
orientation(wk2gt_2,141,113). coords(wk2gt_2,98,62,113).  visible(wk2gt_2,113). happensAt(inactive(wk2gt_2),113). orientation(wk2gt_3,168,113). coords(wk2gt_3,199,203,113).  visible(wk2gt_3,113). happensAt(walking(wk2gt_3),113).
orientation(wk2gt_2,141,114). coords(wk2gt_2,98,62,114).  visible(wk2gt_2,114). happensAt(inactive(wk2gt_2),114). orientation(wk2gt_3,168,114). coords(wk2gt_3,199,201,114).  visible(wk2gt_3,114). happensAt(walking(wk2gt_3),114).
orientation(wk2gt_2,141,115). coords(wk2gt_2,98,62,115).  visible(wk2gt_2,115). happensAt(inactive(wk2gt_2),115). orientation(wk2gt_3,168,115). coords(wk2gt_3,199,200,115).  visible(wk2gt_3,115). happensAt(walking(wk2gt_3),115).
orientation(wk2gt_2,141,116). coords(wk2gt_2,98,62,116).  visible(wk2gt_2,116). happensAt(inactive(wk2gt_2),116). orientation(wk2gt_3,168,116). coords(wk2gt_3,197,198,116).  visible(wk2gt_3,116). happensAt(walking(wk2gt_3),116).
orientation(wk2gt_2,141,117). coords(wk2gt_2,98,62,117).  visible(wk2gt_2,117). happensAt(inactive(wk2gt_2),117). orientation(wk2gt_3,168,117). coords(wk2gt_3,197,194,117).  visible(wk2gt_3,117). happensAt(walking(wk2gt_3),117).
orientation(wk2gt_2,141,118). coords(wk2gt_2,98,62,118).  visible(wk2gt_2,118). happensAt(inactive(wk2gt_2),118). orientation(wk2gt_3,164,118). coords(wk2gt_3,196,193,118).  visible(wk2gt_3,118). happensAt(walking(wk2gt_3),118).
#show holdsAt/2.