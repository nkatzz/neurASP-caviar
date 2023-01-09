
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

	orientation(lb1gt_1,148,96). coords(lb1gt_1,77,117,96).  visible(lb1gt_1,96). happensAt(walking(lb1gt_1),96). orientation(lb1gt_2,145,96). coords(lb1gt_2,89,87,96).  visible(lb1gt_2,96). happensAt(walking(lb1gt_2),96).
orientation(lb1gt_1,150,97). coords(lb1gt_1,77,118,97).  visible(lb1gt_1,97). happensAt(walking(lb1gt_1),97). orientation(lb1gt_2,145,97). coords(lb1gt_2,89,88,97).  visible(lb1gt_2,97). happensAt(walking(lb1gt_2),97).
orientation(lb1gt_1,151,98). coords(lb1gt_1,77,118,98).  visible(lb1gt_1,98). happensAt(walking(lb1gt_1),98). orientation(lb1gt_2,145,98). coords(lb1gt_2,90,88,98).  visible(lb1gt_2,98). happensAt(walking(lb1gt_2),98).
orientation(lb1gt_1,151,99). coords(lb1gt_1,77,118,99).  visible(lb1gt_1,99). happensAt(walking(lb1gt_1),99). orientation(lb1gt_2,145,99). coords(lb1gt_2,90,88,99).  visible(lb1gt_2,99). happensAt(walking(lb1gt_2),99).
orientation(lb1gt_1,151,100). coords(lb1gt_1,77,119,100).  visible(lb1gt_1,100). happensAt(walking(lb1gt_1),100). orientation(lb1gt_2,145,100). coords(lb1gt_2,91,88,100).  visible(lb1gt_2,100). happensAt(walking(lb1gt_2),100).
orientation(lb1gt_1,151,101). coords(lb1gt_1,77,119,101).  visible(lb1gt_1,101). happensAt(walking(lb1gt_1),101). orientation(lb1gt_2,145,101). coords(lb1gt_2,92,88,101).  visible(lb1gt_2,101). happensAt(walking(lb1gt_2),101).
orientation(lb1gt_1,151,102). coords(lb1gt_1,77,119,102).  visible(lb1gt_1,102). happensAt(walking(lb1gt_1),102). orientation(lb1gt_2,145,102). coords(lb1gt_2,92,88,102).  visible(lb1gt_2,102). happensAt(walking(lb1gt_2),102).
orientation(lb1gt_1,151,103). coords(lb1gt_1,77,119,103).  visible(lb1gt_1,103). happensAt(walking(lb1gt_1),103). orientation(lb1gt_2,145,103). coords(lb1gt_2,92,88,103).  visible(lb1gt_2,103). happensAt(walking(lb1gt_2),103).
orientation(lb1gt_1,151,104). coords(lb1gt_1,77,120,104).  visible(lb1gt_1,104). happensAt(walking(lb1gt_1),104). orientation(lb1gt_2,147,104). coords(lb1gt_2,92,89,104).  visible(lb1gt_2,104). happensAt(walking(lb1gt_2),104).
orientation(lb1gt_1,151,105). coords(lb1gt_1,77,120,105).  visible(lb1gt_1,105). happensAt(walking(lb1gt_1),105). orientation(lb1gt_2,147,105). coords(lb1gt_2,93,89,105).  visible(lb1gt_2,105). happensAt(walking(lb1gt_2),105).
orientation(lb1gt_1,151,106). coords(lb1gt_1,78,120,106).  visible(lb1gt_1,106). happensAt(walking(lb1gt_1),106). orientation(lb1gt_2,147,106). coords(lb1gt_2,93,89,106).  visible(lb1gt_2,106). happensAt(walking(lb1gt_2),106).
orientation(lb1gt_1,151,107). coords(lb1gt_1,79,120,107).  visible(lb1gt_1,107). happensAt(walking(lb1gt_1),107). orientation(lb1gt_2,147,107). coords(lb1gt_2,93,89,107).  visible(lb1gt_2,107). happensAt(walking(lb1gt_2),107).
orientation(lb1gt_1,151,108). coords(lb1gt_1,79,120,108).  visible(lb1gt_1,108). happensAt(walking(lb1gt_1),108). orientation(lb1gt_2,147,108). coords(lb1gt_2,93,90,108).  visible(lb1gt_2,108). happensAt(walking(lb1gt_2),108).
orientation(lb1gt_1,151,109). coords(lb1gt_1,80,121,109).  visible(lb1gt_1,109). happensAt(walking(lb1gt_1),109). orientation(lb1gt_2,147,109). coords(lb1gt_2,93,90,109).  visible(lb1gt_2,109). happensAt(walking(lb1gt_2),109).
orientation(lb1gt_1,151,110). coords(lb1gt_1,81,121,110).  visible(lb1gt_1,110). happensAt(walking(lb1gt_1),110). orientation(lb1gt_2,147,110). coords(lb1gt_2,93,90,110).  visible(lb1gt_2,110). happensAt(walking(lb1gt_2),110).
orientation(lb1gt_1,151,111). coords(lb1gt_1,81,122,111).  visible(lb1gt_1,111). happensAt(walking(lb1gt_1),111). orientation(lb1gt_2,147,111). coords(lb1gt_2,93,90,111).  visible(lb1gt_2,111). happensAt(walking(lb1gt_2),111).
orientation(lb1gt_1,151,112). coords(lb1gt_1,81,123,112).  visible(lb1gt_1,112). happensAt(walking(lb1gt_1),112). orientation(lb1gt_2,147,112). coords(lb1gt_2,94,90,112).  visible(lb1gt_2,112). happensAt(walking(lb1gt_2),112).
orientation(lb1gt_1,151,113). coords(lb1gt_1,82,123,113).  visible(lb1gt_1,113). happensAt(walking(lb1gt_1),113). orientation(lb1gt_2,147,113). coords(lb1gt_2,94,91,113).  visible(lb1gt_2,113). happensAt(walking(lb1gt_2),113).
orientation(lb1gt_1,151,114). coords(lb1gt_1,83,123,114).  visible(lb1gt_1,114). happensAt(walking(lb1gt_1),114). orientation(lb1gt_2,147,114). coords(lb1gt_2,94,92,114).  visible(lb1gt_2,114). happensAt(walking(lb1gt_2),114).
orientation(lb1gt_1,151,115). coords(lb1gt_1,83,123,115).  visible(lb1gt_1,115). happensAt(walking(lb1gt_1),115). orientation(lb1gt_2,147,115). coords(lb1gt_2,94,92,115).  visible(lb1gt_2,115). happensAt(walking(lb1gt_2),115).
orientation(lb1gt_1,151,116). coords(lb1gt_1,83,124,116).  visible(lb1gt_1,116). happensAt(walking(lb1gt_1),116). orientation(lb1gt_2,147,116). coords(lb1gt_2,94,94,116).  visible(lb1gt_2,116). happensAt(walking(lb1gt_2),116).
orientation(lb1gt_1,151,117). coords(lb1gt_1,83,124,117).  visible(lb1gt_1,117). happensAt(walking(lb1gt_1),117). orientation(lb1gt_2,147,117). coords(lb1gt_2,94,94,117).  visible(lb1gt_2,117). happensAt(walking(lb1gt_2),117).
orientation(lb1gt_1,151,118). coords(lb1gt_1,83,124,118).  visible(lb1gt_1,118). happensAt(walking(lb1gt_1),118). orientation(lb1gt_2,147,118). coords(lb1gt_2,94,95,118).  visible(lb1gt_2,118). happensAt(walking(lb1gt_2),118).
orientation(lb1gt_1,151,119). coords(lb1gt_1,83,124,119).  visible(lb1gt_1,119). happensAt(walking(lb1gt_1),119). orientation(lb1gt_2,147,119). coords(lb1gt_2,94,96,119).  visible(lb1gt_2,119). happensAt(walking(lb1gt_2),119).
orientation(lb1gt_1,151,120). coords(lb1gt_1,84,124,120).  visible(lb1gt_1,120). happensAt(walking(lb1gt_1),120). orientation(lb1gt_2,147,120). coords(lb1gt_2,94,96,120).  visible(lb1gt_2,120). happensAt(walking(lb1gt_2),120).
#show holdsAt/2.