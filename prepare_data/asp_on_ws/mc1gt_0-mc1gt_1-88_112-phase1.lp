
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

	orientation(mc1gt_0,148,88). coords(mc1gt_0,79,96,88).  visible(mc1gt_0,88). happensAt(walking(mc1gt_0),88). orientation(mc1gt_1,145,88). coords(mc1gt_1,94,83,88).  visible(mc1gt_1,88). happensAt(walking(mc1gt_1),88).
orientation(mc1gt_0,148,89). coords(mc1gt_0,79,96,89).  visible(mc1gt_0,89). happensAt(walking(mc1gt_0),89). orientation(mc1gt_1,145,89). coords(mc1gt_1,94,83,89).  visible(mc1gt_1,89). happensAt(walking(mc1gt_1),89).
orientation(mc1gt_0,148,90). coords(mc1gt_0,79,96,90).  visible(mc1gt_0,90). happensAt(walking(mc1gt_0),90). orientation(mc1gt_1,145,90). coords(mc1gt_1,94,83,90).  visible(mc1gt_1,90). happensAt(walking(mc1gt_1),90).
orientation(mc1gt_0,148,91). coords(mc1gt_0,79,96,91).  visible(mc1gt_0,91). happensAt(walking(mc1gt_0),91). orientation(mc1gt_1,145,91). coords(mc1gt_1,94,83,91).  visible(mc1gt_1,91). happensAt(walking(mc1gt_1),91).
orientation(mc1gt_0,148,92). coords(mc1gt_0,79,96,92).  visible(mc1gt_0,92). happensAt(walking(mc1gt_0),92). orientation(mc1gt_1,145,92). coords(mc1gt_1,94,83,92).  visible(mc1gt_1,92). happensAt(walking(mc1gt_1),92).
orientation(mc1gt_0,148,93). coords(mc1gt_0,79,96,93).  visible(mc1gt_0,93). happensAt(walking(mc1gt_0),93). orientation(mc1gt_1,145,93). coords(mc1gt_1,94,83,93).  visible(mc1gt_1,93). happensAt(walking(mc1gt_1),93).
orientation(mc1gt_0,148,94). coords(mc1gt_0,79,96,94).  visible(mc1gt_0,94). happensAt(walking(mc1gt_0),94). orientation(mc1gt_1,145,94). coords(mc1gt_1,94,83,94).  visible(mc1gt_1,94). happensAt(walking(mc1gt_1),94).
orientation(mc1gt_0,148,95). coords(mc1gt_0,79,96,95).  visible(mc1gt_0,95). happensAt(walking(mc1gt_0),95). orientation(mc1gt_1,145,95). coords(mc1gt_1,94,83,95).  visible(mc1gt_1,95). happensAt(walking(mc1gt_1),95).
orientation(mc1gt_0,148,96). coords(mc1gt_0,79,96,96).  visible(mc1gt_0,96). happensAt(walking(mc1gt_0),96). orientation(mc1gt_1,145,96). coords(mc1gt_1,94,83,96).  visible(mc1gt_1,96). happensAt(walking(mc1gt_1),96).
orientation(mc1gt_0,148,97). coords(mc1gt_0,79,96,97).  visible(mc1gt_0,97). happensAt(walking(mc1gt_0),97). orientation(mc1gt_1,145,97). coords(mc1gt_1,94,83,97).  visible(mc1gt_1,97). happensAt(walking(mc1gt_1),97).
orientation(mc1gt_0,148,98). coords(mc1gt_0,79,96,98).  visible(mc1gt_0,98). happensAt(walking(mc1gt_0),98). orientation(mc1gt_1,145,98). coords(mc1gt_1,94,83,98).  visible(mc1gt_1,98). happensAt(walking(mc1gt_1),98).
orientation(mc1gt_0,148,99). coords(mc1gt_0,79,96,99).  visible(mc1gt_0,99). happensAt(walking(mc1gt_0),99). orientation(mc1gt_1,145,99). coords(mc1gt_1,94,83,99).  visible(mc1gt_1,99). happensAt(walking(mc1gt_1),99).
orientation(mc1gt_0,148,100). coords(mc1gt_0,79,96,100).  visible(mc1gt_0,100). happensAt(walking(mc1gt_0),100). orientation(mc1gt_1,145,100). coords(mc1gt_1,94,83,100).  visible(mc1gt_1,100). happensAt(walking(mc1gt_1),100).
orientation(mc1gt_0,148,101). coords(mc1gt_0,79,97,101).  visible(mc1gt_0,101). happensAt(walking(mc1gt_0),101). orientation(mc1gt_1,145,101). coords(mc1gt_1,94,83,101).  visible(mc1gt_1,101). happensAt(walking(mc1gt_1),101).
orientation(mc1gt_0,148,102). coords(mc1gt_0,79,97,102).  visible(mc1gt_0,102). happensAt(walking(mc1gt_0),102). orientation(mc1gt_1,145,102). coords(mc1gt_1,95,82,102).  visible(mc1gt_1,102). happensAt(walking(mc1gt_1),102).
orientation(mc1gt_0,148,103). coords(mc1gt_0,79,97,103).  visible(mc1gt_0,103). happensAt(walking(mc1gt_0),103). orientation(mc1gt_1,145,103). coords(mc1gt_1,95,82,103).  visible(mc1gt_1,103). happensAt(walking(mc1gt_1),103).
orientation(mc1gt_0,148,104). coords(mc1gt_0,80,97,104).  visible(mc1gt_0,104). happensAt(walking(mc1gt_0),104). orientation(mc1gt_1,145,104). coords(mc1gt_1,95,82,104).  visible(mc1gt_1,104). happensAt(walking(mc1gt_1),104).
orientation(mc1gt_0,148,105). coords(mc1gt_0,80,97,105).  visible(mc1gt_0,105). happensAt(walking(mc1gt_0),105). orientation(mc1gt_1,145,105). coords(mc1gt_1,95,82,105).  visible(mc1gt_1,105). happensAt(walking(mc1gt_1),105).
orientation(mc1gt_0,148,106). coords(mc1gt_0,80,98,106).  visible(mc1gt_0,106). happensAt(walking(mc1gt_0),106). orientation(mc1gt_1,145,106). coords(mc1gt_1,95,82,106).  visible(mc1gt_1,106). happensAt(walking(mc1gt_1),106).
orientation(mc1gt_0,148,107). coords(mc1gt_0,80,99,107).  visible(mc1gt_0,107). happensAt(walking(mc1gt_0),107). orientation(mc1gt_1,145,107). coords(mc1gt_1,95,82,107).  visible(mc1gt_1,107). happensAt(walking(mc1gt_1),107).
orientation(mc1gt_0,148,108). coords(mc1gt_0,80,99,108).  visible(mc1gt_0,108). happensAt(walking(mc1gt_0),108). orientation(mc1gt_1,145,108). coords(mc1gt_1,97,82,108).  visible(mc1gt_1,108). happensAt(walking(mc1gt_1),108).
orientation(mc1gt_0,147,109). coords(mc1gt_0,81,100,109).  visible(mc1gt_0,109). happensAt(walking(mc1gt_0),109). orientation(mc1gt_1,145,109). coords(mc1gt_1,98,82,109).  visible(mc1gt_1,109). happensAt(walking(mc1gt_1),109).
orientation(mc1gt_0,147,110). coords(mc1gt_0,81,101,110).  visible(mc1gt_0,110). happensAt(walking(mc1gt_0),110). orientation(mc1gt_1,145,110). coords(mc1gt_1,99,82,110).  visible(mc1gt_1,110). happensAt(walking(mc1gt_1),110).
orientation(mc1gt_0,147,111). coords(mc1gt_0,82,101,111).  visible(mc1gt_0,111). happensAt(walking(mc1gt_0),111). orientation(mc1gt_1,145,111). coords(mc1gt_1,100,82,111).  visible(mc1gt_1,111). happensAt(walking(mc1gt_1),111).
orientation(mc1gt_0,147,112). coords(mc1gt_0,82,101,112).  visible(mc1gt_0,112). happensAt(walking(mc1gt_0),112). orientation(mc1gt_1,145,112). coords(mc1gt_1,100,82,112).  visible(mc1gt_1,112). happensAt(walking(mc1gt_1),112).
#show holdsAt/2.