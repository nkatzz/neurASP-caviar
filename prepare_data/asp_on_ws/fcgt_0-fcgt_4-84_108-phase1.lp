
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

	orientation(fcgt_0,147,84). coords(fcgt_0,202,154,84).  visible(fcgt_0,84). happensAt(walking(fcgt_0),84). orientation(fcgt_4,165,84). coords(fcgt_4,100,209,84).  visible(fcgt_4,84). happensAt(inactive(fcgt_4),84).
orientation(fcgt_0,147,85). coords(fcgt_0,203,154,85).  visible(fcgt_0,85). happensAt(walking(fcgt_0),85). orientation(fcgt_4,165,85). coords(fcgt_4,101,209,85).  visible(fcgt_4,85). happensAt(inactive(fcgt_4),85).
orientation(fcgt_0,147,86). coords(fcgt_0,204,154,86).  visible(fcgt_0,86). happensAt(walking(fcgt_0),86). orientation(fcgt_4,165,86). coords(fcgt_4,101,209,86).  visible(fcgt_4,86). happensAt(inactive(fcgt_4),86).
orientation(fcgt_0,151,87). coords(fcgt_0,207,155,87).  visible(fcgt_0,87). happensAt(walking(fcgt_0),87). orientation(fcgt_4,165,87). coords(fcgt_4,101,209,87).  visible(fcgt_4,87). happensAt(inactive(fcgt_4),87).
orientation(fcgt_0,151,88). coords(fcgt_0,211,158,88).  visible(fcgt_0,88). happensAt(walking(fcgt_0),88). orientation(fcgt_4,165,88). coords(fcgt_4,101,209,88).  visible(fcgt_4,88). happensAt(inactive(fcgt_4),88).
orientation(fcgt_0,151,89). coords(fcgt_0,213,160,89).  visible(fcgt_0,89). happensAt(walking(fcgt_0),89). orientation(fcgt_4,165,89). coords(fcgt_4,101,209,89).  visible(fcgt_4,89). happensAt(inactive(fcgt_4),89).
orientation(fcgt_0,151,90). coords(fcgt_0,214,162,90).  visible(fcgt_0,90). happensAt(walking(fcgt_0),90). orientation(fcgt_4,165,90). coords(fcgt_4,101,209,90).  visible(fcgt_4,90). happensAt(inactive(fcgt_4),90).
orientation(fcgt_0,151,91). coords(fcgt_0,216,165,91).  visible(fcgt_0,91). happensAt(walking(fcgt_0),91). orientation(fcgt_4,165,91). coords(fcgt_4,101,209,91).  visible(fcgt_4,91). happensAt(inactive(fcgt_4),91).
orientation(fcgt_0,151,92). coords(fcgt_0,216,167,92).  visible(fcgt_0,92). happensAt(walking(fcgt_0),92). orientation(fcgt_4,165,92). coords(fcgt_4,101,210,92).  visible(fcgt_4,92). happensAt(inactive(fcgt_4),92).
orientation(fcgt_0,151,93). coords(fcgt_0,218,169,93).  visible(fcgt_0,93). happensAt(walking(fcgt_0),93). orientation(fcgt_4,165,93). coords(fcgt_4,101,210,93).  visible(fcgt_4,93). happensAt(inactive(fcgt_4),93).
orientation(fcgt_0,151,94). coords(fcgt_0,219,172,94).  visible(fcgt_0,94). happensAt(walking(fcgt_0),94). orientation(fcgt_4,165,94). coords(fcgt_4,101,210,94).  visible(fcgt_4,94). happensAt(inactive(fcgt_4),94).
orientation(fcgt_0,151,95). coords(fcgt_0,223,174,95).  visible(fcgt_0,95). happensAt(walking(fcgt_0),95). orientation(fcgt_4,165,95). coords(fcgt_4,101,210,95).  visible(fcgt_4,95). happensAt(inactive(fcgt_4),95).
orientation(fcgt_0,154,96). coords(fcgt_0,226,177,96).  visible(fcgt_0,96). happensAt(walking(fcgt_0),96). orientation(fcgt_4,165,96). coords(fcgt_4,101,210,96).  visible(fcgt_4,96). happensAt(inactive(fcgt_4),96).
orientation(fcgt_0,160,97). coords(fcgt_0,229,179,97).  visible(fcgt_0,97). happensAt(walking(fcgt_0),97). orientation(fcgt_4,165,97). coords(fcgt_4,101,210,97).  visible(fcgt_4,97). happensAt(inactive(fcgt_4),97).
orientation(fcgt_0,162,98). coords(fcgt_0,231,181,98).  visible(fcgt_0,98). happensAt(walking(fcgt_0),98). orientation(fcgt_4,165,98). coords(fcgt_4,101,210,98).  visible(fcgt_4,98). happensAt(inactive(fcgt_4),98).
orientation(fcgt_0,166,99). coords(fcgt_0,231,185,99).  visible(fcgt_0,99). happensAt(walking(fcgt_0),99). orientation(fcgt_4,165,99). coords(fcgt_4,101,210,99).  visible(fcgt_4,99). happensAt(inactive(fcgt_4),99).
orientation(fcgt_0,174,100). coords(fcgt_0,231,185,100).  visible(fcgt_0,100). happensAt(walking(fcgt_0),100). orientation(fcgt_4,165,100). coords(fcgt_4,101,210,100).  visible(fcgt_4,100). happensAt(inactive(fcgt_4),100).
orientation(fcgt_0,177,101). coords(fcgt_0,233,187,101).  visible(fcgt_0,101). happensAt(walking(fcgt_0),101). orientation(fcgt_4,165,101). coords(fcgt_4,101,210,101).  visible(fcgt_4,101). happensAt(inactive(fcgt_4),101).
orientation(fcgt_0,179,102). coords(fcgt_0,233,190,102).  visible(fcgt_0,102). happensAt(walking(fcgt_0),102). orientation(fcgt_4,165,102). coords(fcgt_4,101,210,102).  visible(fcgt_4,102). happensAt(inactive(fcgt_4),102).
orientation(fcgt_0,178,103). coords(fcgt_0,233,193,103).  visible(fcgt_0,103). happensAt(walking(fcgt_0),103). orientation(fcgt_4,165,103). coords(fcgt_4,101,210,103).  visible(fcgt_4,103). happensAt(inactive(fcgt_4),103).
orientation(fcgt_0,177,104). coords(fcgt_0,232,196,104).  visible(fcgt_0,104). happensAt(walking(fcgt_0),104). orientation(fcgt_4,165,104). coords(fcgt_4,101,210,104).  visible(fcgt_4,104). happensAt(inactive(fcgt_4),104).
orientation(fcgt_0,177,105). coords(fcgt_0,232,198,105).  visible(fcgt_0,105). happensAt(walking(fcgt_0),105). orientation(fcgt_4,165,105). coords(fcgt_4,101,210,105).  visible(fcgt_4,105). happensAt(inactive(fcgt_4),105).
orientation(fcgt_0,176,106). coords(fcgt_0,235,201,106).  visible(fcgt_0,106). happensAt(walking(fcgt_0),106). orientation(fcgt_4,165,106). coords(fcgt_4,101,210,106).  visible(fcgt_4,106). happensAt(inactive(fcgt_4),106).
orientation(fcgt_0,174,107). coords(fcgt_0,237,203,107).  visible(fcgt_0,107). happensAt(walking(fcgt_0),107). orientation(fcgt_4,165,107). coords(fcgt_4,101,210,107).  visible(fcgt_4,107). happensAt(inactive(fcgt_4),107).
orientation(fcgt_0,175,108). coords(fcgt_0,240,206,108).  visible(fcgt_0,108). happensAt(walking(fcgt_0),108). orientation(fcgt_4,165,108). coords(fcgt_4,101,210,108).  visible(fcgt_4,108). happensAt(inactive(fcgt_4),108).
#show holdsAt/2.