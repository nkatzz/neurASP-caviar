
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

	orientation(lbgt_0,154,84). coords(lbgt_0,152,108,84).  visible(lbgt_0,84). happensAt(walking(lbgt_0),84). orientation(lbgt_1,145,84). coords(lbgt_1,79,68,84).  visible(lbgt_1,84). happensAt(walking(lbgt_1),84).
orientation(lbgt_0,154,85). coords(lbgt_0,152,108,85).  visible(lbgt_0,85). happensAt(walking(lbgt_0),85). orientation(lbgt_1,145,85). coords(lbgt_1,78,68,85).  visible(lbgt_1,85). happensAt(walking(lbgt_1),85).
orientation(lbgt_0,154,86). coords(lbgt_0,152,108,86).  visible(lbgt_0,86). happensAt(walking(lbgt_0),86). orientation(lbgt_1,145,86). coords(lbgt_1,78,68,86).  visible(lbgt_1,86). happensAt(walking(lbgt_1),86).
orientation(lbgt_0,154,87). coords(lbgt_0,152,108,87).  visible(lbgt_0,87). happensAt(walking(lbgt_0),87). orientation(lbgt_1,145,87). coords(lbgt_1,78,68,87).  visible(lbgt_1,87). happensAt(walking(lbgt_1),87).
orientation(lbgt_0,154,88). coords(lbgt_0,152,108,88).  visible(lbgt_0,88). happensAt(walking(lbgt_0),88). orientation(lbgt_1,145,88). coords(lbgt_1,77,68,88).  visible(lbgt_1,88). happensAt(walking(lbgt_1),88).
orientation(lbgt_0,154,89). coords(lbgt_0,152,108,89).  visible(lbgt_0,89). happensAt(walking(lbgt_0),89). orientation(lbgt_1,145,89). coords(lbgt_1,77,68,89).  visible(lbgt_1,89). happensAt(walking(lbgt_1),89).
orientation(lbgt_0,154,90). coords(lbgt_0,151,108,90).  visible(lbgt_0,90). happensAt(walking(lbgt_0),90). orientation(lbgt_1,145,90). coords(lbgt_1,77,68,90).  visible(lbgt_1,90). happensAt(walking(lbgt_1),90).
orientation(lbgt_0,154,91). coords(lbgt_0,151,108,91).  visible(lbgt_0,91). happensAt(inactive(lbgt_0),91). orientation(lbgt_1,145,91). coords(lbgt_1,77,71,91).  visible(lbgt_1,91). happensAt(walking(lbgt_1),91).
orientation(lbgt_0,154,92). coords(lbgt_0,151,108,92).  visible(lbgt_0,92). happensAt(inactive(lbgt_0),92). orientation(lbgt_1,145,92). coords(lbgt_1,77,72,92).  visible(lbgt_1,92). happensAt(walking(lbgt_1),92).
orientation(lbgt_0,154,93). coords(lbgt_0,151,108,93).  visible(lbgt_0,93). happensAt(inactive(lbgt_0),93). orientation(lbgt_1,145,93). coords(lbgt_1,76,72,93).  visible(lbgt_1,93). happensAt(walking(lbgt_1),93).
orientation(lbgt_0,154,94). coords(lbgt_0,151,108,94).  visible(lbgt_0,94). happensAt(inactive(lbgt_0),94). orientation(lbgt_1,145,94). coords(lbgt_1,76,72,94).  visible(lbgt_1,94). happensAt(walking(lbgt_1),94).
orientation(lbgt_0,154,95). coords(lbgt_0,151,108,95).  visible(lbgt_0,95). happensAt(inactive(lbgt_0),95). orientation(lbgt_1,145,95). coords(lbgt_1,76,73,95).  visible(lbgt_1,95). happensAt(walking(lbgt_1),95).
orientation(lbgt_0,154,96). coords(lbgt_0,151,108,96).  visible(lbgt_0,96). happensAt(inactive(lbgt_0),96). orientation(lbgt_1,145,96). coords(lbgt_1,76,73,96).  visible(lbgt_1,96). happensAt(walking(lbgt_1),96).
orientation(lbgt_0,154,97). coords(lbgt_0,152,108,97).  visible(lbgt_0,97). happensAt(inactive(lbgt_0),97). orientation(lbgt_1,145,97). coords(lbgt_1,76,73,97).  visible(lbgt_1,97). happensAt(walking(lbgt_1),97).
orientation(lbgt_0,154,98). coords(lbgt_0,152,108,98).  visible(lbgt_0,98). happensAt(inactive(lbgt_0),98). orientation(lbgt_1,145,98). coords(lbgt_1,76,74,98).  visible(lbgt_1,98). happensAt(walking(lbgt_1),98).
orientation(lbgt_0,154,99). coords(lbgt_0,152,108,99).  visible(lbgt_0,99). happensAt(inactive(lbgt_0),99). orientation(lbgt_1,145,99). coords(lbgt_1,76,75,99).  visible(lbgt_1,99). happensAt(walking(lbgt_1),99).
orientation(lbgt_0,154,100). coords(lbgt_0,152,108,100).  visible(lbgt_0,100). happensAt(inactive(lbgt_0),100). orientation(lbgt_1,145,100). coords(lbgt_1,76,75,100).  visible(lbgt_1,100). happensAt(walking(lbgt_1),100).
orientation(lbgt_0,154,101). coords(lbgt_0,152,108,101).  visible(lbgt_0,101). happensAt(inactive(lbgt_0),101). orientation(lbgt_1,145,101). coords(lbgt_1,76,75,101).  visible(lbgt_1,101). happensAt(walking(lbgt_1),101).
orientation(lbgt_0,154,102). coords(lbgt_0,152,108,102).  visible(lbgt_0,102). happensAt(inactive(lbgt_0),102). orientation(lbgt_1,145,102). coords(lbgt_1,76,76,102).  visible(lbgt_1,102). happensAt(walking(lbgt_1),102).
orientation(lbgt_0,154,103). coords(lbgt_0,152,108,103).  visible(lbgt_0,103). happensAt(inactive(lbgt_0),103). orientation(lbgt_1,145,103). coords(lbgt_1,76,76,103).  visible(lbgt_1,103). happensAt(walking(lbgt_1),103).
orientation(lbgt_0,154,104). coords(lbgt_0,152,108,104).  visible(lbgt_0,104). happensAt(inactive(lbgt_0),104). orientation(lbgt_1,145,104). coords(lbgt_1,76,76,104).  visible(lbgt_1,104). happensAt(walking(lbgt_1),104).
orientation(lbgt_0,154,105). coords(lbgt_0,152,108,105).  visible(lbgt_0,105). happensAt(inactive(lbgt_0),105). orientation(lbgt_1,145,105). coords(lbgt_1,76,78,105).  visible(lbgt_1,105). happensAt(walking(lbgt_1),105).
orientation(lbgt_0,154,106). coords(lbgt_0,152,108,106).  visible(lbgt_0,106). happensAt(inactive(lbgt_0),106). orientation(lbgt_1,145,106). coords(lbgt_1,76,79,106).  visible(lbgt_1,106). happensAt(walking(lbgt_1),106).
orientation(lbgt_0,154,107). coords(lbgt_0,152,108,107).  visible(lbgt_0,107). happensAt(inactive(lbgt_0),107). orientation(lbgt_1,145,107). coords(lbgt_1,76,79,107).  visible(lbgt_1,107). happensAt(walking(lbgt_1),107).
orientation(lbgt_0,154,108). coords(lbgt_0,153,108,108).  visible(lbgt_0,108). happensAt(inactive(lbgt_0),108). orientation(lbgt_1,145,108). coords(lbgt_1,76,79,108).  visible(lbgt_1,108). happensAt(walking(lbgt_1),108).
#show holdsAt/2.