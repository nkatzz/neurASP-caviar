
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

	orientation(fcgt_3,159,96). coords(fcgt_3,103,163,96).  visible(fcgt_3,96). happensAt(inactive(fcgt_3),96). orientation(fcgt_4,165,96). coords(fcgt_4,101,210,96).  visible(fcgt_4,96). happensAt(inactive(fcgt_4),96).
orientation(fcgt_3,159,97). coords(fcgt_3,103,163,97).  visible(fcgt_3,97). happensAt(inactive(fcgt_3),97). orientation(fcgt_4,165,97). coords(fcgt_4,101,210,97).  visible(fcgt_4,97). happensAt(inactive(fcgt_4),97).
orientation(fcgt_3,159,98). coords(fcgt_3,103,163,98).  visible(fcgt_3,98). happensAt(inactive(fcgt_3),98). orientation(fcgt_4,165,98). coords(fcgt_4,101,210,98).  visible(fcgt_4,98). happensAt(inactive(fcgt_4),98).
orientation(fcgt_3,159,99). coords(fcgt_3,103,163,99).  visible(fcgt_3,99). happensAt(inactive(fcgt_3),99). orientation(fcgt_4,165,99). coords(fcgt_4,101,210,99).  visible(fcgt_4,99). happensAt(inactive(fcgt_4),99).
orientation(fcgt_3,159,100). coords(fcgt_3,103,163,100).  visible(fcgt_3,100). happensAt(inactive(fcgt_3),100). orientation(fcgt_4,165,100). coords(fcgt_4,101,210,100).  visible(fcgt_4,100). happensAt(inactive(fcgt_4),100).
orientation(fcgt_3,159,101). coords(fcgt_3,103,163,101).  visible(fcgt_3,101). happensAt(inactive(fcgt_3),101). orientation(fcgt_4,165,101). coords(fcgt_4,101,210,101).  visible(fcgt_4,101). happensAt(inactive(fcgt_4),101).
orientation(fcgt_3,159,102). coords(fcgt_3,103,163,102).  visible(fcgt_3,102). happensAt(inactive(fcgt_3),102). orientation(fcgt_4,165,102). coords(fcgt_4,101,210,102).  visible(fcgt_4,102). happensAt(inactive(fcgt_4),102).
orientation(fcgt_3,159,103). coords(fcgt_3,103,163,103).  visible(fcgt_3,103). happensAt(inactive(fcgt_3),103). orientation(fcgt_4,165,103). coords(fcgt_4,101,210,103).  visible(fcgt_4,103). happensAt(inactive(fcgt_4),103).
orientation(fcgt_3,159,104). coords(fcgt_3,103,163,104).  visible(fcgt_3,104). happensAt(inactive(fcgt_3),104). orientation(fcgt_4,165,104). coords(fcgt_4,101,210,104).  visible(fcgt_4,104). happensAt(inactive(fcgt_4),104).
orientation(fcgt_3,159,105). coords(fcgt_3,103,163,105).  visible(fcgt_3,105). happensAt(inactive(fcgt_3),105). orientation(fcgt_4,165,105). coords(fcgt_4,101,210,105).  visible(fcgt_4,105). happensAt(inactive(fcgt_4),105).
orientation(fcgt_3,159,106). coords(fcgt_3,103,163,106).  visible(fcgt_3,106). happensAt(inactive(fcgt_3),106). orientation(fcgt_4,165,106). coords(fcgt_4,101,210,106).  visible(fcgt_4,106). happensAt(inactive(fcgt_4),106).
orientation(fcgt_3,159,107). coords(fcgt_3,103,163,107).  visible(fcgt_3,107). happensAt(inactive(fcgt_3),107). orientation(fcgt_4,165,107). coords(fcgt_4,101,210,107).  visible(fcgt_4,107). happensAt(inactive(fcgt_4),107).
orientation(fcgt_3,159,108). coords(fcgt_3,103,163,108).  visible(fcgt_3,108). happensAt(inactive(fcgt_3),108). orientation(fcgt_4,165,108). coords(fcgt_4,101,210,108).  visible(fcgt_4,108). happensAt(inactive(fcgt_4),108).
orientation(fcgt_3,159,109). coords(fcgt_3,103,163,109).  visible(fcgt_3,109). happensAt(inactive(fcgt_3),109). orientation(fcgt_4,165,109). coords(fcgt_4,101,210,109).  visible(fcgt_4,109). happensAt(inactive(fcgt_4),109).
orientation(fcgt_3,159,110). coords(fcgt_3,103,163,110).  visible(fcgt_3,110). happensAt(inactive(fcgt_3),110). orientation(fcgt_4,165,110). coords(fcgt_4,101,210,110).  visible(fcgt_4,110). happensAt(inactive(fcgt_4),110).
orientation(fcgt_3,159,111). coords(fcgt_3,103,163,111).  visible(fcgt_3,111). happensAt(inactive(fcgt_3),111). orientation(fcgt_4,165,111). coords(fcgt_4,101,210,111).  visible(fcgt_4,111). happensAt(inactive(fcgt_4),111).
orientation(fcgt_3,159,112). coords(fcgt_3,103,163,112).  visible(fcgt_3,112). happensAt(inactive(fcgt_3),112). orientation(fcgt_4,165,112). coords(fcgt_4,101,210,112).  visible(fcgt_4,112). happensAt(inactive(fcgt_4),112).
orientation(fcgt_3,159,113). coords(fcgt_3,103,163,113).  visible(fcgt_3,113). happensAt(inactive(fcgt_3),113). orientation(fcgt_4,165,113). coords(fcgt_4,101,210,113).  visible(fcgt_4,113). happensAt(inactive(fcgt_4),113).
orientation(fcgt_3,159,114). coords(fcgt_3,103,163,114).  visible(fcgt_3,114). happensAt(inactive(fcgt_3),114). orientation(fcgt_4,165,114). coords(fcgt_4,101,210,114).  visible(fcgt_4,114). happensAt(inactive(fcgt_4),114).
orientation(fcgt_3,159,115). coords(fcgt_3,103,163,115).  visible(fcgt_3,115). happensAt(inactive(fcgt_3),115). orientation(fcgt_4,165,115). coords(fcgt_4,101,210,115).  visible(fcgt_4,115). happensAt(inactive(fcgt_4),115).
orientation(fcgt_3,159,116). coords(fcgt_3,103,163,116).  visible(fcgt_3,116). happensAt(inactive(fcgt_3),116). orientation(fcgt_4,165,116). coords(fcgt_4,101,210,116).  visible(fcgt_4,116). happensAt(inactive(fcgt_4),116).
orientation(fcgt_3,159,117). coords(fcgt_3,103,163,117).  visible(fcgt_3,117). happensAt(inactive(fcgt_3),117). orientation(fcgt_4,165,117). coords(fcgt_4,101,210,117).  visible(fcgt_4,117). happensAt(inactive(fcgt_4),117).
orientation(fcgt_3,159,118). coords(fcgt_3,103,163,118).  visible(fcgt_3,118). happensAt(inactive(fcgt_3),118). orientation(fcgt_4,165,118). coords(fcgt_4,101,210,118).  visible(fcgt_4,118). happensAt(inactive(fcgt_4),118).
orientation(fcgt_3,159,119). coords(fcgt_3,103,163,119).  visible(fcgt_3,119). happensAt(inactive(fcgt_3),119). orientation(fcgt_4,165,119). coords(fcgt_4,101,210,119).  visible(fcgt_4,119). happensAt(inactive(fcgt_4),119).
orientation(fcgt_3,159,120). coords(fcgt_3,103,163,120).  visible(fcgt_3,120). happensAt(inactive(fcgt_3),120). orientation(fcgt_4,165,120). coords(fcgt_4,101,210,120).  visible(fcgt_4,120). happensAt(inactive(fcgt_4),120).
#show holdsAt/2.