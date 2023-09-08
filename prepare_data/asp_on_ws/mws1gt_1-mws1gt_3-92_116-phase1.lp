
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

	orientation(mws1gt_1,160,92). coords(mws1gt_1,88,181,92).  visible(mws1gt_1,92). happensAt(walking(mws1gt_1),92). orientation(mws1gt_3,24,92). coords(mws1gt_3,248,244,92).  visible(mws1gt_3,92). happensAt(walking(mws1gt_3),92).
orientation(mws1gt_1,160,93). coords(mws1gt_1,89,180,93).  visible(mws1gt_1,93). happensAt(walking(mws1gt_1),93). orientation(mws1gt_3,24,93). coords(mws1gt_3,246,243,93).  visible(mws1gt_3,93). happensAt(walking(mws1gt_3),93).
orientation(mws1gt_1,160,94). coords(mws1gt_1,90,180,94).  visible(mws1gt_1,94). happensAt(walking(mws1gt_1),94). orientation(mws1gt_3,24,94). coords(mws1gt_3,246,241,94).  visible(mws1gt_3,94). happensAt(walking(mws1gt_3),94).
orientation(mws1gt_1,160,95). coords(mws1gt_1,90,179,95).  visible(mws1gt_1,95). happensAt(walking(mws1gt_1),95). orientation(mws1gt_3,19,95). coords(mws1gt_3,244,240,95).  visible(mws1gt_3,95). happensAt(walking(mws1gt_3),95).
orientation(mws1gt_1,160,96). coords(mws1gt_1,90,178,96).  visible(mws1gt_1,96). happensAt(walking(mws1gt_1),96). orientation(mws1gt_3,10,96). coords(mws1gt_3,242,237,96).  visible(mws1gt_3,96). happensAt(walking(mws1gt_3),96).
orientation(mws1gt_1,160,97). coords(mws1gt_1,91,177,97).  visible(mws1gt_1,97). happensAt(walking(mws1gt_1),97). orientation(mws1gt_3,10,97). coords(mws1gt_3,242,236,97).  visible(mws1gt_3,97). happensAt(walking(mws1gt_3),97).
orientation(mws1gt_1,160,98). coords(mws1gt_1,91,177,98).  visible(mws1gt_1,98). happensAt(walking(mws1gt_1),98). orientation(mws1gt_3,11,98). coords(mws1gt_3,242,234,98).  visible(mws1gt_3,98). happensAt(walking(mws1gt_3),98).
orientation(mws1gt_1,160,99). coords(mws1gt_1,91,177,99).  visible(mws1gt_1,99). happensAt(walking(mws1gt_1),99). orientation(mws1gt_3,6,99). coords(mws1gt_3,241,231,99).  visible(mws1gt_3,99). happensAt(walking(mws1gt_3),99).
orientation(mws1gt_1,155,100). coords(mws1gt_1,91,176,100).  visible(mws1gt_1,100). happensAt(walking(mws1gt_1),100). orientation(mws1gt_3,3,100). coords(mws1gt_3,241,229,100).  visible(mws1gt_3,100). happensAt(walking(mws1gt_3),100).
orientation(mws1gt_1,155,101). coords(mws1gt_1,91,176,101).  visible(mws1gt_1,101). happensAt(walking(mws1gt_1),101). orientation(mws1gt_3,171,101). coords(mws1gt_3,240,226,101).  visible(mws1gt_3,101). happensAt(walking(mws1gt_3),101).
orientation(mws1gt_1,155,102). coords(mws1gt_1,91,175,102).  visible(mws1gt_1,102). happensAt(walking(mws1gt_1),102). orientation(mws1gt_3,4,102). coords(mws1gt_3,239,223,102).  visible(mws1gt_3,102). happensAt(walking(mws1gt_3),102).
orientation(mws1gt_1,155,103). coords(mws1gt_1,94,174,103).  visible(mws1gt_1,103). happensAt(walking(mws1gt_1),103). orientation(mws1gt_3,4,103). coords(mws1gt_3,239,222,103).  visible(mws1gt_3,103). happensAt(walking(mws1gt_3),103).
orientation(mws1gt_1,155,104). coords(mws1gt_1,96,174,104).  visible(mws1gt_1,104). happensAt(walking(mws1gt_1),104). orientation(mws1gt_3,174,104). coords(mws1gt_3,238,221,104).  visible(mws1gt_3,104). happensAt(walking(mws1gt_3),104).
orientation(mws1gt_1,155,105). coords(mws1gt_1,98,173,105).  visible(mws1gt_1,105). happensAt(walking(mws1gt_1),105). orientation(mws1gt_3,176,105). coords(mws1gt_3,237,221,105).  visible(mws1gt_3,105). happensAt(walking(mws1gt_3),105).
orientation(mws1gt_1,155,106). coords(mws1gt_1,98,173,106).  visible(mws1gt_1,106). happensAt(walking(mws1gt_1),106). orientation(mws1gt_3,169,106). coords(mws1gt_3,236,221,106).  visible(mws1gt_3,106). happensAt(walking(mws1gt_3),106).
orientation(mws1gt_1,155,107). coords(mws1gt_1,99,173,107).  visible(mws1gt_1,107). happensAt(walking(mws1gt_1),107). orientation(mws1gt_3,172,107). coords(mws1gt_3,235,220,107).  visible(mws1gt_3,107). happensAt(walking(mws1gt_3),107).
orientation(mws1gt_1,155,108). coords(mws1gt_1,100,173,108).  visible(mws1gt_1,108). happensAt(walking(mws1gt_1),108). orientation(mws1gt_3,176,108). coords(mws1gt_3,234,218,108).  visible(mws1gt_3,108). happensAt(walking(mws1gt_3),108).
orientation(mws1gt_1,155,109). coords(mws1gt_1,102,172,109).  visible(mws1gt_1,109). happensAt(walking(mws1gt_1),109). orientation(mws1gt_3,173,109). coords(mws1gt_3,232,216,109).  visible(mws1gt_3,109). happensAt(walking(mws1gt_3),109).
orientation(mws1gt_1,155,110). coords(mws1gt_1,102,172,110).  visible(mws1gt_1,110). happensAt(walking(mws1gt_1),110). orientation(mws1gt_3,170,110). coords(mws1gt_3,231,214,110).  visible(mws1gt_3,110). happensAt(walking(mws1gt_3),110).
orientation(mws1gt_1,155,111). coords(mws1gt_1,102,172,111).  visible(mws1gt_1,111). happensAt(walking(mws1gt_1),111). orientation(mws1gt_3,178,111). coords(mws1gt_3,228,211,111).  visible(mws1gt_3,111). happensAt(walking(mws1gt_3),111).
orientation(mws1gt_1,155,112). coords(mws1gt_1,102,171,112).  visible(mws1gt_1,112). happensAt(walking(mws1gt_1),112). orientation(mws1gt_3,178,112). coords(mws1gt_3,226,209,112).  visible(mws1gt_3,112). happensAt(walking(mws1gt_3),112).
orientation(mws1gt_1,155,113). coords(mws1gt_1,102,171,113).  visible(mws1gt_1,113). happensAt(walking(mws1gt_1),113). orientation(mws1gt_3,176,113). coords(mws1gt_3,224,206,113).  visible(mws1gt_3,113). happensAt(walking(mws1gt_3),113).
orientation(mws1gt_1,155,114). coords(mws1gt_1,102,171,114).  visible(mws1gt_1,114). happensAt(walking(mws1gt_1),114). orientation(mws1gt_3,178,114). coords(mws1gt_3,222,205,114).  visible(mws1gt_3,114). happensAt(walking(mws1gt_3),114).
orientation(mws1gt_1,155,115). coords(mws1gt_1,102,170,115).  visible(mws1gt_1,115). happensAt(walking(mws1gt_1),115). orientation(mws1gt_3,170,115). coords(mws1gt_3,221,203,115).  visible(mws1gt_3,115). happensAt(walking(mws1gt_3),115).
orientation(mws1gt_1,155,116). coords(mws1gt_1,103,169,116).  visible(mws1gt_1,116). happensAt(walking(mws1gt_1),116). orientation(mws1gt_3,160,116). coords(mws1gt_3,220,201,116).  visible(mws1gt_3,116). happensAt(walking(mws1gt_3),116).
#show holdsAt/2.