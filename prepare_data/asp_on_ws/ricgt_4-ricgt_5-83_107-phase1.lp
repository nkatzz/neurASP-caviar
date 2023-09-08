
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

	orientation(ricgt_4,172,83). coords(ricgt_4,27,133,83).  visible(ricgt_4,83). happensAt(walking(ricgt_4),83). orientation(ricgt_5,0,83). coords(ricgt_5,27,125,83).  visible(ricgt_5,83). happensAt(walking(ricgt_5),83).
orientation(ricgt_4,172,84). coords(ricgt_4,27,133,84).  visible(ricgt_4,84). happensAt(walking(ricgt_4),84). orientation(ricgt_5,0,84). coords(ricgt_5,27,127,84).  visible(ricgt_5,84). happensAt(walking(ricgt_5),84).
orientation(ricgt_4,172,85). coords(ricgt_4,27,133,85).  visible(ricgt_4,85). happensAt(walking(ricgt_4),85). orientation(ricgt_5,0,85). coords(ricgt_5,27,126,85).  visible(ricgt_5,85). happensAt(walking(ricgt_5),85).
orientation(ricgt_4,172,86). coords(ricgt_4,27,133,86).  visible(ricgt_4,86). happensAt(walking(ricgt_4),86). orientation(ricgt_5,0,86). coords(ricgt_5,27,126,86).  visible(ricgt_5,86). happensAt(walking(ricgt_5),86).
orientation(ricgt_4,172,87). coords(ricgt_4,27,133,87).  visible(ricgt_4,87). happensAt(walking(ricgt_4),87). orientation(ricgt_5,0,87). coords(ricgt_5,27,127,87).  visible(ricgt_5,87). happensAt(walking(ricgt_5),87).
orientation(ricgt_4,172,88). coords(ricgt_4,27,133,88).  visible(ricgt_4,88). happensAt(walking(ricgt_4),88). orientation(ricgt_5,0,88). coords(ricgt_5,27,127,88).  visible(ricgt_5,88). happensAt(walking(ricgt_5),88).
orientation(ricgt_4,172,89). coords(ricgt_4,27,133,89).  visible(ricgt_4,89). happensAt(walking(ricgt_4),89). orientation(ricgt_5,0,89). coords(ricgt_5,27,127,89).  visible(ricgt_5,89). happensAt(walking(ricgt_5),89).
orientation(ricgt_4,172,90). coords(ricgt_4,27,133,90).  visible(ricgt_4,90). happensAt(walking(ricgt_4),90). orientation(ricgt_5,0,90). coords(ricgt_5,27,127,90).  visible(ricgt_5,90). happensAt(walking(ricgt_5),90).
orientation(ricgt_4,172,91). coords(ricgt_4,27,133,91).  visible(ricgt_4,91). happensAt(walking(ricgt_4),91). orientation(ricgt_5,0,91). coords(ricgt_5,27,127,91).  visible(ricgt_5,91). happensAt(walking(ricgt_5),91).
orientation(ricgt_4,172,92). coords(ricgt_4,27,133,92).  visible(ricgt_4,92). happensAt(walking(ricgt_4),92). orientation(ricgt_5,0,92). coords(ricgt_5,27,126,92).  visible(ricgt_5,92). happensAt(walking(ricgt_5),92).
orientation(ricgt_4,172,93). coords(ricgt_4,26,133,93).  visible(ricgt_4,93). happensAt(walking(ricgt_4),93). orientation(ricgt_5,0,93). coords(ricgt_5,27,126,93).  visible(ricgt_5,93). happensAt(walking(ricgt_5),93).
orientation(ricgt_4,172,94). coords(ricgt_4,26,134,94).  visible(ricgt_4,94). happensAt(walking(ricgt_4),94). orientation(ricgt_5,0,94). coords(ricgt_5,27,126,94).  visible(ricgt_5,94). happensAt(walking(ricgt_5),94).
orientation(ricgt_4,172,95). coords(ricgt_4,26,134,95).  visible(ricgt_4,95). happensAt(walking(ricgt_4),95). orientation(ricgt_5,0,95). coords(ricgt_5,27,126,95).  visible(ricgt_5,95). happensAt(walking(ricgt_5),95).
orientation(ricgt_4,172,96). coords(ricgt_4,26,134,96).  visible(ricgt_4,96). happensAt(walking(ricgt_4),96). orientation(ricgt_5,0,96). coords(ricgt_5,27,126,96).  visible(ricgt_5,96). happensAt(walking(ricgt_5),96).
orientation(ricgt_4,172,97). coords(ricgt_4,26,134,97).  visible(ricgt_4,97). happensAt(walking(ricgt_4),97). orientation(ricgt_5,0,97). coords(ricgt_5,27,126,97).  visible(ricgt_5,97). happensAt(walking(ricgt_5),97).
orientation(ricgt_4,172,98). coords(ricgt_4,26,134,98).  visible(ricgt_4,98). happensAt(walking(ricgt_4),98). orientation(ricgt_5,0,98). coords(ricgt_5,27,126,98).  visible(ricgt_5,98). happensAt(walking(ricgt_5),98).
orientation(ricgt_4,172,99). coords(ricgt_4,26,134,99).  visible(ricgt_4,99). happensAt(walking(ricgt_4),99). orientation(ricgt_5,0,99). coords(ricgt_5,27,126,99).  visible(ricgt_5,99). happensAt(walking(ricgt_5),99).
orientation(ricgt_4,172,100). coords(ricgt_4,26,134,100).  visible(ricgt_4,100). happensAt(walking(ricgt_4),100). orientation(ricgt_5,0,100). coords(ricgt_5,27,126,100).  visible(ricgt_5,100). happensAt(walking(ricgt_5),100).
orientation(ricgt_4,172,101). coords(ricgt_4,26,134,101).  visible(ricgt_4,101). happensAt(walking(ricgt_4),101). orientation(ricgt_5,0,101). coords(ricgt_5,27,126,101).  visible(ricgt_5,101). happensAt(walking(ricgt_5),101).
orientation(ricgt_4,172,102). coords(ricgt_4,26,134,102).  visible(ricgt_4,102). happensAt(walking(ricgt_4),102). orientation(ricgt_5,0,102). coords(ricgt_5,27,126,102).  visible(ricgt_5,102). happensAt(walking(ricgt_5),102).
orientation(ricgt_4,172,103). coords(ricgt_4,26,134,103).  visible(ricgt_4,103). happensAt(walking(ricgt_4),103). orientation(ricgt_5,0,103). coords(ricgt_5,27,127,103).  visible(ricgt_5,103). happensAt(walking(ricgt_5),103).
orientation(ricgt_4,172,104). coords(ricgt_4,26,134,104).  visible(ricgt_4,104). happensAt(walking(ricgt_4),104). orientation(ricgt_5,0,104). coords(ricgt_5,27,127,104).  visible(ricgt_5,104). happensAt(walking(ricgt_5),104).
orientation(ricgt_4,172,105). coords(ricgt_4,26,134,105).  visible(ricgt_4,105). happensAt(walking(ricgt_4),105). orientation(ricgt_5,0,105). coords(ricgt_5,27,127,105).  visible(ricgt_5,105). happensAt(walking(ricgt_5),105).
orientation(ricgt_4,172,106). coords(ricgt_4,26,134,106).  visible(ricgt_4,106). happensAt(walking(ricgt_4),106). orientation(ricgt_5,0,106). coords(ricgt_5,27,127,106).  visible(ricgt_5,106). happensAt(walking(ricgt_5),106).
orientation(ricgt_4,172,107). coords(ricgt_4,26,134,107).  visible(ricgt_4,107). happensAt(walking(ricgt_4),107). orientation(ricgt_5,0,107). coords(ricgt_5,27,127,107).  visible(ricgt_5,107). happensAt(walking(ricgt_5),107).
#show holdsAt/2.