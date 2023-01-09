
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

	orientation(br1gt_0,159,96). coords(br1gt_0,183,140,96).  visible(br1gt_0,96). happensAt(inactive(br1gt_0),96). orientation(br1gt_2,171,96). coords(br1gt_2,86,207,96).  visible(br1gt_2,96). happensAt(inactive(br1gt_2),96).
orientation(br1gt_0,159,97). coords(br1gt_0,183,140,97).  visible(br1gt_0,97). happensAt(inactive(br1gt_0),97). orientation(br1gt_2,171,97). coords(br1gt_2,86,207,97).  visible(br1gt_2,97). happensAt(inactive(br1gt_2),97).
orientation(br1gt_0,159,98). coords(br1gt_0,183,140,98).  visible(br1gt_0,98). happensAt(inactive(br1gt_0),98). orientation(br1gt_2,171,98). coords(br1gt_2,86,207,98).  visible(br1gt_2,98). happensAt(inactive(br1gt_2),98).
orientation(br1gt_0,159,99). coords(br1gt_0,183,140,99).  visible(br1gt_0,99). happensAt(inactive(br1gt_0),99). orientation(br1gt_2,171,99). coords(br1gt_2,86,207,99).  visible(br1gt_2,99). happensAt(inactive(br1gt_2),99).
orientation(br1gt_0,159,100). coords(br1gt_0,183,140,100).  visible(br1gt_0,100). happensAt(inactive(br1gt_0),100). orientation(br1gt_2,171,100). coords(br1gt_2,86,207,100).  visible(br1gt_2,100). happensAt(inactive(br1gt_2),100).
orientation(br1gt_0,159,101). coords(br1gt_0,182,140,101).  visible(br1gt_0,101). happensAt(inactive(br1gt_0),101). orientation(br1gt_2,171,101). coords(br1gt_2,86,207,101).  visible(br1gt_2,101). happensAt(inactive(br1gt_2),101).
orientation(br1gt_0,159,102). coords(br1gt_0,182,140,102).  visible(br1gt_0,102). happensAt(inactive(br1gt_0),102). orientation(br1gt_2,171,102). coords(br1gt_2,86,207,102).  visible(br1gt_2,102). happensAt(inactive(br1gt_2),102).
orientation(br1gt_0,159,103). coords(br1gt_0,182,140,103).  visible(br1gt_0,103). happensAt(inactive(br1gt_0),103). orientation(br1gt_2,171,103). coords(br1gt_2,86,207,103).  visible(br1gt_2,103). happensAt(inactive(br1gt_2),103).
orientation(br1gt_0,159,104). coords(br1gt_0,182,140,104).  visible(br1gt_0,104). happensAt(inactive(br1gt_0),104). orientation(br1gt_2,171,104). coords(br1gt_2,86,207,104).  visible(br1gt_2,104). happensAt(inactive(br1gt_2),104).
orientation(br1gt_0,155,105). coords(br1gt_0,182,140,105).  visible(br1gt_0,105). happensAt(inactive(br1gt_0),105). orientation(br1gt_2,171,105). coords(br1gt_2,86,207,105).  visible(br1gt_2,105). happensAt(inactive(br1gt_2),105).
orientation(br1gt_0,159,106). coords(br1gt_0,183,141,106).  visible(br1gt_0,106). happensAt(inactive(br1gt_0),106). orientation(br1gt_2,171,106). coords(br1gt_2,86,207,106).  visible(br1gt_2,106). happensAt(inactive(br1gt_2),106).
orientation(br1gt_0,159,107). coords(br1gt_0,183,141,107).  visible(br1gt_0,107). happensAt(inactive(br1gt_0),107). orientation(br1gt_2,171,107). coords(br1gt_2,86,207,107).  visible(br1gt_2,107). happensAt(inactive(br1gt_2),107).
orientation(br1gt_0,159,108). coords(br1gt_0,183,141,108).  visible(br1gt_0,108). happensAt(inactive(br1gt_0),108). orientation(br1gt_2,171,108). coords(br1gt_2,86,207,108).  visible(br1gt_2,108). happensAt(inactive(br1gt_2),108).
orientation(br1gt_0,153,109). coords(br1gt_0,184,141,109).  visible(br1gt_0,109). happensAt(walking(br1gt_0),109). orientation(br1gt_2,171,109). coords(br1gt_2,86,207,109).  visible(br1gt_2,109). happensAt(inactive(br1gt_2),109).
orientation(br1gt_0,153,110). coords(br1gt_0,184,142,110).  visible(br1gt_0,110). happensAt(walking(br1gt_0),110). orientation(br1gt_2,171,110). coords(br1gt_2,86,207,110).  visible(br1gt_2,110). happensAt(inactive(br1gt_2),110).
orientation(br1gt_0,154,111). coords(br1gt_0,185,142,111).  visible(br1gt_0,111). happensAt(walking(br1gt_0),111). orientation(br1gt_2,171,111). coords(br1gt_2,86,207,111).  visible(br1gt_2,111). happensAt(inactive(br1gt_2),111).
orientation(br1gt_0,159,112). coords(br1gt_0,186,143,112).  visible(br1gt_0,112). happensAt(walking(br1gt_0),112). orientation(br1gt_2,171,112). coords(br1gt_2,86,207,112).  visible(br1gt_2,112). happensAt(inactive(br1gt_2),112).
orientation(br1gt_0,152,113). coords(br1gt_0,186,143,113).  visible(br1gt_0,113). happensAt(walking(br1gt_0),113). orientation(br1gt_2,171,113). coords(br1gt_2,86,207,113).  visible(br1gt_2,113). happensAt(inactive(br1gt_2),113).
orientation(br1gt_0,153,114). coords(br1gt_0,188,144,114).  visible(br1gt_0,114). happensAt(walking(br1gt_0),114). orientation(br1gt_2,171,114). coords(br1gt_2,86,207,114).  visible(br1gt_2,114). happensAt(inactive(br1gt_2),114).
orientation(br1gt_0,154,115). coords(br1gt_0,190,145,115).  visible(br1gt_0,115). happensAt(walking(br1gt_0),115). orientation(br1gt_2,171,115). coords(br1gt_2,86,207,115).  visible(br1gt_2,115). happensAt(inactive(br1gt_2),115).
orientation(br1gt_0,153,116). coords(br1gt_0,191,145,116).  visible(br1gt_0,116). happensAt(walking(br1gt_0),116). orientation(br1gt_2,171,116). coords(br1gt_2,86,207,116).  visible(br1gt_2,116). happensAt(inactive(br1gt_2),116).
orientation(br1gt_0,159,117). coords(br1gt_0,193,146,117).  visible(br1gt_0,117). happensAt(walking(br1gt_0),117). orientation(br1gt_2,171,117). coords(br1gt_2,86,207,117).  visible(br1gt_2,117). happensAt(inactive(br1gt_2),117).
orientation(br1gt_0,151,118). coords(br1gt_0,195,146,118).  visible(br1gt_0,118). happensAt(walking(br1gt_0),118). orientation(br1gt_2,171,118). coords(br1gt_2,86,207,118).  visible(br1gt_2,118). happensAt(inactive(br1gt_2),118).
orientation(br1gt_0,154,119). coords(br1gt_0,195,148,119).  visible(br1gt_0,119). happensAt(walking(br1gt_0),119). orientation(br1gt_2,171,119). coords(br1gt_2,86,207,119).  visible(br1gt_2,119). happensAt(inactive(br1gt_2),119).
orientation(br1gt_0,152,120). coords(br1gt_0,197,149,120).  visible(br1gt_0,120). happensAt(walking(br1gt_0),120). orientation(br1gt_2,171,120). coords(br1gt_2,86,207,120).  visible(br1gt_2,120). happensAt(inactive(br1gt_2),120).
#show holdsAt/2.