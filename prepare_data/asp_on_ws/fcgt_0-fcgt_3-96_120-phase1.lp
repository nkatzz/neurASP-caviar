
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

	orientation(fcgt_0,154,96). coords(fcgt_0,226,177,96).  visible(fcgt_0,96). happensAt(walking(fcgt_0),96). orientation(fcgt_3,159,96). coords(fcgt_3,103,163,96).  visible(fcgt_3,96). happensAt(inactive(fcgt_3),96).
orientation(fcgt_0,160,97). coords(fcgt_0,229,179,97).  visible(fcgt_0,97). happensAt(walking(fcgt_0),97). orientation(fcgt_3,159,97). coords(fcgt_3,103,163,97).  visible(fcgt_3,97). happensAt(inactive(fcgt_3),97).
orientation(fcgt_0,162,98). coords(fcgt_0,231,181,98).  visible(fcgt_0,98). happensAt(walking(fcgt_0),98). orientation(fcgt_3,159,98). coords(fcgt_3,103,163,98).  visible(fcgt_3,98). happensAt(inactive(fcgt_3),98).
orientation(fcgt_0,166,99). coords(fcgt_0,231,185,99).  visible(fcgt_0,99). happensAt(walking(fcgt_0),99). orientation(fcgt_3,159,99). coords(fcgt_3,103,163,99).  visible(fcgt_3,99). happensAt(inactive(fcgt_3),99).
orientation(fcgt_0,174,100). coords(fcgt_0,231,185,100).  visible(fcgt_0,100). happensAt(walking(fcgt_0),100). orientation(fcgt_3,159,100). coords(fcgt_3,103,163,100).  visible(fcgt_3,100). happensAt(inactive(fcgt_3),100).
orientation(fcgt_0,177,101). coords(fcgt_0,233,187,101).  visible(fcgt_0,101). happensAt(walking(fcgt_0),101). orientation(fcgt_3,159,101). coords(fcgt_3,103,163,101).  visible(fcgt_3,101). happensAt(inactive(fcgt_3),101).
orientation(fcgt_0,179,102). coords(fcgt_0,233,190,102).  visible(fcgt_0,102). happensAt(walking(fcgt_0),102). orientation(fcgt_3,159,102). coords(fcgt_3,103,163,102).  visible(fcgt_3,102). happensAt(inactive(fcgt_3),102).
orientation(fcgt_0,178,103). coords(fcgt_0,233,193,103).  visible(fcgt_0,103). happensAt(walking(fcgt_0),103). orientation(fcgt_3,159,103). coords(fcgt_3,103,163,103).  visible(fcgt_3,103). happensAt(inactive(fcgt_3),103).
orientation(fcgt_0,177,104). coords(fcgt_0,232,196,104).  visible(fcgt_0,104). happensAt(walking(fcgt_0),104). orientation(fcgt_3,159,104). coords(fcgt_3,103,163,104).  visible(fcgt_3,104). happensAt(inactive(fcgt_3),104).
orientation(fcgt_0,177,105). coords(fcgt_0,232,198,105).  visible(fcgt_0,105). happensAt(walking(fcgt_0),105). orientation(fcgt_3,159,105). coords(fcgt_3,103,163,105).  visible(fcgt_3,105). happensAt(inactive(fcgt_3),105).
orientation(fcgt_0,176,106). coords(fcgt_0,235,201,106).  visible(fcgt_0,106). happensAt(walking(fcgt_0),106). orientation(fcgt_3,159,106). coords(fcgt_3,103,163,106).  visible(fcgt_3,106). happensAt(inactive(fcgt_3),106).
orientation(fcgt_0,174,107). coords(fcgt_0,237,203,107).  visible(fcgt_0,107). happensAt(walking(fcgt_0),107). orientation(fcgt_3,159,107). coords(fcgt_3,103,163,107).  visible(fcgt_3,107). happensAt(inactive(fcgt_3),107).
orientation(fcgt_0,175,108). coords(fcgt_0,240,206,108).  visible(fcgt_0,108). happensAt(walking(fcgt_0),108). orientation(fcgt_3,159,108). coords(fcgt_3,103,163,108).  visible(fcgt_3,108). happensAt(inactive(fcgt_3),108).
orientation(fcgt_0,173,109). coords(fcgt_0,241,207,109).  visible(fcgt_0,109). happensAt(walking(fcgt_0),109). orientation(fcgt_3,159,109). coords(fcgt_3,103,163,109).  visible(fcgt_3,109). happensAt(inactive(fcgt_3),109).
orientation(fcgt_0,176,110). coords(fcgt_0,242,209,110).  visible(fcgt_0,110). happensAt(walking(fcgt_0),110). orientation(fcgt_3,159,110). coords(fcgt_3,103,163,110).  visible(fcgt_3,110). happensAt(inactive(fcgt_3),110).
orientation(fcgt_0,178,111). coords(fcgt_0,243,211,111).  visible(fcgt_0,111). happensAt(walking(fcgt_0),111). orientation(fcgt_3,159,111). coords(fcgt_3,103,163,111).  visible(fcgt_3,111). happensAt(inactive(fcgt_3),111).
orientation(fcgt_0,0,112). coords(fcgt_0,244,216,112).  visible(fcgt_0,112). happensAt(walking(fcgt_0),112). orientation(fcgt_3,159,112). coords(fcgt_3,103,163,112).  visible(fcgt_3,112). happensAt(inactive(fcgt_3),112).
orientation(fcgt_0,174,113). coords(fcgt_0,245,220,113).  visible(fcgt_0,113). happensAt(walking(fcgt_0),113). orientation(fcgt_3,159,113). coords(fcgt_3,103,163,113).  visible(fcgt_3,113). happensAt(inactive(fcgt_3),113).
orientation(fcgt_0,170,114). coords(fcgt_0,245,223,114).  visible(fcgt_0,114). happensAt(walking(fcgt_0),114). orientation(fcgt_3,159,114). coords(fcgt_3,103,163,114).  visible(fcgt_3,114). happensAt(inactive(fcgt_3),114).
orientation(fcgt_0,164,115). coords(fcgt_0,246,227,115).  visible(fcgt_0,115). happensAt(walking(fcgt_0),115). orientation(fcgt_3,159,115). coords(fcgt_3,103,163,115).  visible(fcgt_3,115). happensAt(inactive(fcgt_3),115).
orientation(fcgt_0,160,116). coords(fcgt_0,247,229,116).  visible(fcgt_0,116). happensAt(walking(fcgt_0),116). orientation(fcgt_3,159,116). coords(fcgt_3,103,163,116).  visible(fcgt_3,116). happensAt(inactive(fcgt_3),116).
orientation(fcgt_0,163,117). coords(fcgt_0,248,231,117).  visible(fcgt_0,117). happensAt(walking(fcgt_0),117). orientation(fcgt_3,159,117). coords(fcgt_3,103,163,117).  visible(fcgt_3,117). happensAt(inactive(fcgt_3),117).
orientation(fcgt_0,158,118). coords(fcgt_0,249,233,118).  visible(fcgt_0,118). happensAt(walking(fcgt_0),118). orientation(fcgt_3,159,118). coords(fcgt_3,103,163,118).  visible(fcgt_3,118). happensAt(inactive(fcgt_3),118).
orientation(fcgt_0,151,119). coords(fcgt_0,250,236,119).  visible(fcgt_0,119). happensAt(walking(fcgt_0),119). orientation(fcgt_3,159,119). coords(fcgt_3,103,163,119).  visible(fcgt_3,119). happensAt(inactive(fcgt_3),119).
orientation(fcgt_0,152,120). coords(fcgt_0,253,236,120).  visible(fcgt_0,120). happensAt(walking(fcgt_0),120). orientation(fcgt_3,159,120). coords(fcgt_3,103,163,120).  visible(fcgt_3,120). happensAt(inactive(fcgt_3),120).
#show holdsAt/2.