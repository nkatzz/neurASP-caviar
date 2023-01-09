
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

	orientation(lb2gt_0,141,84). coords(lb2gt_0,193,124,84).  visible(lb2gt_0,84). happensAt(active(lb2gt_0),84). orientation(lb2gt_3,146,84). coords(lb2gt_3,13,49,84).  visible(lb2gt_3,84). happensAt(walking(lb2gt_3),84).
orientation(lb2gt_0,141,85). coords(lb2gt_0,193,124,85).  visible(lb2gt_0,85). happensAt(active(lb2gt_0),85). orientation(lb2gt_3,146,85). coords(lb2gt_3,12,49,85).  visible(lb2gt_3,85). happensAt(walking(lb2gt_3),85).
orientation(lb2gt_0,141,86). coords(lb2gt_0,193,124,86).  visible(lb2gt_0,86). happensAt(active(lb2gt_0),86). orientation(lb2gt_3,146,86). coords(lb2gt_3,13,49,86).  visible(lb2gt_3,86). happensAt(walking(lb2gt_3),86).
orientation(lb2gt_0,141,87). coords(lb2gt_0,193,124,87).  visible(lb2gt_0,87). happensAt(active(lb2gt_0),87). orientation(lb2gt_3,146,87). coords(lb2gt_3,13,49,87).  visible(lb2gt_3,87). happensAt(walking(lb2gt_3),87).
orientation(lb2gt_0,141,88). coords(lb2gt_0,193,124,88).  visible(lb2gt_0,88). happensAt(active(lb2gt_0),88). orientation(lb2gt_3,146,88). coords(lb2gt_3,13,49,88).  visible(lb2gt_3,88). happensAt(walking(lb2gt_3),88).
orientation(lb2gt_0,141,89). coords(lb2gt_0,193,124,89).  visible(lb2gt_0,89). happensAt(active(lb2gt_0),89). orientation(lb2gt_3,146,89). coords(lb2gt_3,13,50,89).  visible(lb2gt_3,89). happensAt(walking(lb2gt_3),89).
orientation(lb2gt_0,141,90). coords(lb2gt_0,193,124,90).  visible(lb2gt_0,90). happensAt(active(lb2gt_0),90). orientation(lb2gt_3,146,90). coords(lb2gt_3,12,50,90).  visible(lb2gt_3,90). happensAt(walking(lb2gt_3),90).
orientation(lb2gt_0,141,91). coords(lb2gt_0,193,124,91).  visible(lb2gt_0,91). happensAt(active(lb2gt_0),91). orientation(lb2gt_3,146,91). coords(lb2gt_3,12,50,91).  visible(lb2gt_3,91). happensAt(walking(lb2gt_3),91).
orientation(lb2gt_0,141,92). coords(lb2gt_0,193,124,92).  visible(lb2gt_0,92). happensAt(active(lb2gt_0),92). orientation(lb2gt_3,146,92). coords(lb2gt_3,12,50,92).  visible(lb2gt_3,92). happensAt(walking(lb2gt_3),92).
orientation(lb2gt_0,141,93). coords(lb2gt_0,193,123,93).  visible(lb2gt_0,93). happensAt(active(lb2gt_0),93). orientation(lb2gt_3,146,93). coords(lb2gt_3,12,50,93).  visible(lb2gt_3,93). happensAt(walking(lb2gt_3),93).
orientation(lb2gt_0,141,94). coords(lb2gt_0,193,123,94).  visible(lb2gt_0,94). happensAt(active(lb2gt_0),94). orientation(lb2gt_3,146,94). coords(lb2gt_3,12,50,94).  visible(lb2gt_3,94). happensAt(walking(lb2gt_3),94).
orientation(lb2gt_0,141,95). coords(lb2gt_0,193,123,95).  visible(lb2gt_0,95). happensAt(active(lb2gt_0),95). orientation(lb2gt_3,146,95). coords(lb2gt_3,12,50,95).  visible(lb2gt_3,95). happensAt(walking(lb2gt_3),95).
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
#show holdsAt/2.