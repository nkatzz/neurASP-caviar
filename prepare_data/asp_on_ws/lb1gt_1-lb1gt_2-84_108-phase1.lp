
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

	orientation(lb1gt_1,147,84). coords(lb1gt_1,75,109,84).  visible(lb1gt_1,84). happensAt(walking(lb1gt_1),84). orientation(lb1gt_2,142,84). coords(lb1gt_2,88,84,84).  visible(lb1gt_2,84). happensAt(walking(lb1gt_2),84).
orientation(lb1gt_1,147,85). coords(lb1gt_1,75,109,85).  visible(lb1gt_1,85). happensAt(walking(lb1gt_1),85). orientation(lb1gt_2,142,85). coords(lb1gt_2,88,85,85).  visible(lb1gt_2,85). happensAt(walking(lb1gt_2),85).
orientation(lb1gt_1,147,86). coords(lb1gt_1,75,109,86).  visible(lb1gt_1,86). happensAt(walking(lb1gt_1),86). orientation(lb1gt_2,142,86). coords(lb1gt_2,88,85,86).  visible(lb1gt_2,86). happensAt(walking(lb1gt_2),86).
orientation(lb1gt_1,147,87). coords(lb1gt_1,75,110,87).  visible(lb1gt_1,87). happensAt(walking(lb1gt_1),87). orientation(lb1gt_2,142,87). coords(lb1gt_2,88,86,87).  visible(lb1gt_2,87). happensAt(walking(lb1gt_2),87).
orientation(lb1gt_1,147,88). coords(lb1gt_1,75,110,88).  visible(lb1gt_1,88). happensAt(walking(lb1gt_1),88). orientation(lb1gt_2,142,88). coords(lb1gt_2,88,86,88).  visible(lb1gt_2,88). happensAt(walking(lb1gt_2),88).
orientation(lb1gt_1,147,89). coords(lb1gt_1,76,111,89).  visible(lb1gt_1,89). happensAt(walking(lb1gt_1),89). orientation(lb1gt_2,142,89). coords(lb1gt_2,88,86,89).  visible(lb1gt_2,89). happensAt(walking(lb1gt_2),89).
orientation(lb1gt_1,147,90). coords(lb1gt_1,76,113,90).  visible(lb1gt_1,90). happensAt(walking(lb1gt_1),90). orientation(lb1gt_2,142,90). coords(lb1gt_2,89,87,90).  visible(lb1gt_2,90). happensAt(walking(lb1gt_2),90).
orientation(lb1gt_1,147,91). coords(lb1gt_1,76,114,91).  visible(lb1gt_1,91). happensAt(walking(lb1gt_1),91). orientation(lb1gt_2,142,91). coords(lb1gt_2,89,87,91).  visible(lb1gt_2,91). happensAt(walking(lb1gt_2),91).
orientation(lb1gt_1,147,92). coords(lb1gt_1,76,115,92).  visible(lb1gt_1,92). happensAt(walking(lb1gt_1),92). orientation(lb1gt_2,142,92). coords(lb1gt_2,89,87,92).  visible(lb1gt_2,92). happensAt(walking(lb1gt_2),92).
orientation(lb1gt_1,147,93). coords(lb1gt_1,76,116,93).  visible(lb1gt_1,93). happensAt(walking(lb1gt_1),93). orientation(lb1gt_2,145,93). coords(lb1gt_2,89,87,93).  visible(lb1gt_2,93). happensAt(walking(lb1gt_2),93).
orientation(lb1gt_1,148,94). coords(lb1gt_1,77,116,94).  visible(lb1gt_1,94). happensAt(walking(lb1gt_1),94). orientation(lb1gt_2,145,94). coords(lb1gt_2,89,87,94).  visible(lb1gt_2,94). happensAt(walking(lb1gt_2),94).
orientation(lb1gt_1,148,95). coords(lb1gt_1,77,116,95).  visible(lb1gt_1,95). happensAt(walking(lb1gt_1),95). orientation(lb1gt_2,144,95). coords(lb1gt_2,89,87,95).  visible(lb1gt_2,95). happensAt(walking(lb1gt_2),95).
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
#show holdsAt/2.