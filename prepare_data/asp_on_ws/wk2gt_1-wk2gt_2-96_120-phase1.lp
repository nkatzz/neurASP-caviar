
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

	orientation(wk2gt_1,138,96). coords(wk2gt_1,115,53,96).  visible(wk2gt_1,96). happensAt(inactive(wk2gt_1),96). orientation(wk2gt_2,141,96). coords(wk2gt_2,98,62,96).  visible(wk2gt_2,96). happensAt(inactive(wk2gt_2),96).
orientation(wk2gt_1,139,97). coords(wk2gt_1,115,53,97).  visible(wk2gt_1,97). happensAt(inactive(wk2gt_1),97). orientation(wk2gt_2,141,97). coords(wk2gt_2,98,62,97).  visible(wk2gt_2,97). happensAt(inactive(wk2gt_2),97).
orientation(wk2gt_1,136,98). coords(wk2gt_1,115,53,98).  visible(wk2gt_1,98). happensAt(inactive(wk2gt_1),98). orientation(wk2gt_2,141,98). coords(wk2gt_2,98,62,98).  visible(wk2gt_2,98). happensAt(inactive(wk2gt_2),98).
orientation(wk2gt_1,136,99). coords(wk2gt_1,115,52,99).  visible(wk2gt_1,99). happensAt(inactive(wk2gt_1),99). orientation(wk2gt_2,141,99). coords(wk2gt_2,98,62,99).  visible(wk2gt_2,99). happensAt(inactive(wk2gt_2),99).
orientation(wk2gt_1,135,100). coords(wk2gt_1,115,52,100).  visible(wk2gt_1,100). happensAt(inactive(wk2gt_1),100). orientation(wk2gt_2,141,100). coords(wk2gt_2,98,62,100).  visible(wk2gt_2,100). happensAt(inactive(wk2gt_2),100).
orientation(wk2gt_1,135,101). coords(wk2gt_1,115,52,101).  visible(wk2gt_1,101). happensAt(inactive(wk2gt_1),101). orientation(wk2gt_2,141,101). coords(wk2gt_2,98,62,101).  visible(wk2gt_2,101). happensAt(inactive(wk2gt_2),101).
orientation(wk2gt_1,138,102). coords(wk2gt_1,115,52,102).  visible(wk2gt_1,102). happensAt(inactive(wk2gt_1),102). orientation(wk2gt_2,141,102). coords(wk2gt_2,98,62,102).  visible(wk2gt_2,102). happensAt(inactive(wk2gt_2),102).
orientation(wk2gt_1,139,103). coords(wk2gt_1,115,52,103).  visible(wk2gt_1,103). happensAt(inactive(wk2gt_1),103). orientation(wk2gt_2,141,103). coords(wk2gt_2,98,62,103).  visible(wk2gt_2,103). happensAt(inactive(wk2gt_2),103).
orientation(wk2gt_1,135,104). coords(wk2gt_1,115,52,104).  visible(wk2gt_1,104). happensAt(inactive(wk2gt_1),104). orientation(wk2gt_2,141,104). coords(wk2gt_2,98,62,104).  visible(wk2gt_2,104). happensAt(inactive(wk2gt_2),104).
orientation(wk2gt_1,135,105). coords(wk2gt_1,115,52,105).  visible(wk2gt_1,105). happensAt(inactive(wk2gt_1),105). orientation(wk2gt_2,141,105). coords(wk2gt_2,98,62,105).  visible(wk2gt_2,105). happensAt(inactive(wk2gt_2),105).
orientation(wk2gt_1,134,106). coords(wk2gt_1,115,52,106).  visible(wk2gt_1,106). happensAt(inactive(wk2gt_1),106). orientation(wk2gt_2,141,106). coords(wk2gt_2,98,62,106).  visible(wk2gt_2,106). happensAt(inactive(wk2gt_2),106).
orientation(wk2gt_1,136,107). coords(wk2gt_1,115,52,107).  visible(wk2gt_1,107). happensAt(inactive(wk2gt_1),107). orientation(wk2gt_2,141,107). coords(wk2gt_2,98,62,107).  visible(wk2gt_2,107). happensAt(inactive(wk2gt_2),107).
orientation(wk2gt_1,137,108). coords(wk2gt_1,115,52,108).  visible(wk2gt_1,108). happensAt(inactive(wk2gt_1),108). orientation(wk2gt_2,141,108). coords(wk2gt_2,98,62,108).  visible(wk2gt_2,108). happensAt(inactive(wk2gt_2),108).
orientation(wk2gt_1,134,109). coords(wk2gt_1,115,52,109).  visible(wk2gt_1,109). happensAt(inactive(wk2gt_1),109). orientation(wk2gt_2,141,109). coords(wk2gt_2,98,62,109).  visible(wk2gt_2,109). happensAt(inactive(wk2gt_2),109).
orientation(wk2gt_1,136,110). coords(wk2gt_1,115,52,110).  visible(wk2gt_1,110). happensAt(inactive(wk2gt_1),110). orientation(wk2gt_2,141,110). coords(wk2gt_2,98,62,110).  visible(wk2gt_2,110). happensAt(inactive(wk2gt_2),110).
orientation(wk2gt_1,134,111). coords(wk2gt_1,115,52,111).  visible(wk2gt_1,111). happensAt(inactive(wk2gt_1),111). orientation(wk2gt_2,141,111). coords(wk2gt_2,98,62,111).  visible(wk2gt_2,111). happensAt(inactive(wk2gt_2),111).
orientation(wk2gt_1,136,112). coords(wk2gt_1,115,52,112).  visible(wk2gt_1,112). happensAt(inactive(wk2gt_1),112). orientation(wk2gt_2,141,112). coords(wk2gt_2,98,62,112).  visible(wk2gt_2,112). happensAt(inactive(wk2gt_2),112).
orientation(wk2gt_1,139,113). coords(wk2gt_1,115,52,113).  visible(wk2gt_1,113). happensAt(inactive(wk2gt_1),113). orientation(wk2gt_2,141,113). coords(wk2gt_2,98,62,113).  visible(wk2gt_2,113). happensAt(inactive(wk2gt_2),113).
orientation(wk2gt_1,136,114). coords(wk2gt_1,115,52,114).  visible(wk2gt_1,114). happensAt(inactive(wk2gt_1),114). orientation(wk2gt_2,141,114). coords(wk2gt_2,98,62,114).  visible(wk2gt_2,114). happensAt(inactive(wk2gt_2),114).
orientation(wk2gt_1,135,115). coords(wk2gt_1,115,52,115).  visible(wk2gt_1,115). happensAt(inactive(wk2gt_1),115). orientation(wk2gt_2,141,115). coords(wk2gt_2,98,62,115).  visible(wk2gt_2,115). happensAt(inactive(wk2gt_2),115).
orientation(wk2gt_1,136,116). coords(wk2gt_1,115,52,116).  visible(wk2gt_1,116). happensAt(inactive(wk2gt_1),116). orientation(wk2gt_2,141,116). coords(wk2gt_2,98,62,116).  visible(wk2gt_2,116). happensAt(inactive(wk2gt_2),116).
orientation(wk2gt_1,135,117). coords(wk2gt_1,115,52,117).  visible(wk2gt_1,117). happensAt(inactive(wk2gt_1),117). orientation(wk2gt_2,141,117). coords(wk2gt_2,98,62,117).  visible(wk2gt_2,117). happensAt(inactive(wk2gt_2),117).
orientation(wk2gt_1,136,118). coords(wk2gt_1,115,52,118).  visible(wk2gt_1,118). happensAt(inactive(wk2gt_1),118). orientation(wk2gt_2,141,118). coords(wk2gt_2,98,62,118).  visible(wk2gt_2,118). happensAt(inactive(wk2gt_2),118).
orientation(wk2gt_1,136,119). coords(wk2gt_1,115,52,119).  visible(wk2gt_1,119). happensAt(inactive(wk2gt_1),119). orientation(wk2gt_2,141,119). coords(wk2gt_2,98,62,119).  visible(wk2gt_2,119). happensAt(inactive(wk2gt_2),119).
orientation(wk2gt_1,132,120). coords(wk2gt_1,114,52,120).  visible(wk2gt_1,120). happensAt(inactive(wk2gt_1),120). orientation(wk2gt_2,141,120). coords(wk2gt_2,98,62,120).  visible(wk2gt_2,120). happensAt(inactive(wk2gt_2),120).
#show holdsAt/2.