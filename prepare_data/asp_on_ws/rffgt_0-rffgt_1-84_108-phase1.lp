
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

	orientation(rffgt_0,138,84). coords(rffgt_0,135,53,84).  visible(rffgt_0,84). happensAt(walking(rffgt_0),84). orientation(rffgt_1,143,84). coords(rffgt_1,156,109,84).  visible(rffgt_1,84). happensAt(active(rffgt_1),84).
orientation(rffgt_0,137,85). coords(rffgt_0,136,53,85).  visible(rffgt_0,85). happensAt(walking(rffgt_0),85). orientation(rffgt_1,143,85). coords(rffgt_1,156,106,85).  visible(rffgt_1,85). happensAt(active(rffgt_1),85).
orientation(rffgt_0,139,86). coords(rffgt_0,137,53,86).  visible(rffgt_0,86). happensAt(walking(rffgt_0),86). orientation(rffgt_1,143,86). coords(rffgt_1,156,104,86).  visible(rffgt_1,86). happensAt(active(rffgt_1),86).
orientation(rffgt_0,140,87). coords(rffgt_0,137,52,87).  visible(rffgt_0,87). happensAt(walking(rffgt_0),87). orientation(rffgt_1,143,87). coords(rffgt_1,156,103,87).  visible(rffgt_1,87). happensAt(active(rffgt_1),87).
orientation(rffgt_0,140,88). coords(rffgt_0,137,52,88).  visible(rffgt_0,88). happensAt(walking(rffgt_0),88). orientation(rffgt_1,143,88). coords(rffgt_1,155,102,88).  visible(rffgt_1,88). happensAt(active(rffgt_1),88).
orientation(rffgt_0,140,89). coords(rffgt_0,137,52,89).  visible(rffgt_0,89). happensAt(walking(rffgt_0),89). orientation(rffgt_1,143,89). coords(rffgt_1,153,102,89).  visible(rffgt_1,89). happensAt(active(rffgt_1),89).
orientation(rffgt_0,140,90). coords(rffgt_0,137,51,90).  visible(rffgt_0,90). happensAt(walking(rffgt_0),90). orientation(rffgt_1,145,90). coords(rffgt_1,152,103,90).  visible(rffgt_1,90). happensAt(active(rffgt_1),90).
orientation(rffgt_0,135,91). coords(rffgt_0,138,51,91).  visible(rffgt_0,91). happensAt(walking(rffgt_0),91). orientation(rffgt_1,145,91). coords(rffgt_1,151,103,91).  visible(rffgt_1,91). happensAt(active(rffgt_1),91).
orientation(rffgt_0,135,92). coords(rffgt_0,138,51,92).  visible(rffgt_0,92). happensAt(walking(rffgt_0),92). orientation(rffgt_1,145,92). coords(rffgt_1,151,104,92).  visible(rffgt_1,92). happensAt(active(rffgt_1),92).
orientation(rffgt_0,135,93). coords(rffgt_0,138,49,93).  visible(rffgt_0,93). happensAt(walking(rffgt_0),93). orientation(rffgt_1,145,93). coords(rffgt_1,150,105,93).  visible(rffgt_1,93). happensAt(active(rffgt_1),93).
orientation(rffgt_0,135,94). coords(rffgt_0,138,49,94).  visible(rffgt_0,94). happensAt(walking(rffgt_0),94). orientation(rffgt_1,146,94). coords(rffgt_1,149,106,94).  visible(rffgt_1,94). happensAt(active(rffgt_1),94).
orientation(rffgt_0,135,95). coords(rffgt_0,138,47,95).  visible(rffgt_0,95). happensAt(walking(rffgt_0),95). orientation(rffgt_1,146,95). coords(rffgt_1,149,106,95).  visible(rffgt_1,95). happensAt(active(rffgt_1),95).
orientation(rffgt_0,135,96). coords(rffgt_0,139,46,96).  visible(rffgt_0,96). happensAt(walking(rffgt_0),96). orientation(rffgt_1,146,96). coords(rffgt_1,150,106,96).  visible(rffgt_1,96). happensAt(active(rffgt_1),96).
orientation(rffgt_0,135,97). coords(rffgt_0,139,46,97).  visible(rffgt_0,97). happensAt(walking(rffgt_0),97). orientation(rffgt_1,146,97). coords(rffgt_1,150,106,97).  visible(rffgt_1,97). happensAt(active(rffgt_1),97).
orientation(rffgt_0,135,98). coords(rffgt_0,139,44,98).  visible(rffgt_0,98). happensAt(walking(rffgt_0),98). orientation(rffgt_1,146,98). coords(rffgt_1,150,107,98).  visible(rffgt_1,98). happensAt(active(rffgt_1),98).
orientation(rffgt_0,135,99). coords(rffgt_0,139,44,99).  visible(rffgt_0,99). happensAt(walking(rffgt_0),99). orientation(rffgt_1,146,99). coords(rffgt_1,151,107,99).  visible(rffgt_1,99). happensAt(active(rffgt_1),99).
orientation(rffgt_0,135,100). coords(rffgt_0,139,43,100).  visible(rffgt_0,100). happensAt(walking(rffgt_0),100). orientation(rffgt_1,144,100). coords(rffgt_1,151,108,100).  visible(rffgt_1,100). happensAt(active(rffgt_1),100).
orientation(rffgt_0,135,101). coords(rffgt_0,139,43,101).  visible(rffgt_0,101). happensAt(walking(rffgt_0),101). orientation(rffgt_1,147,101). coords(rffgt_1,152,108,101).  visible(rffgt_1,101). happensAt(active(rffgt_1),101).
orientation(rffgt_0,135,102). coords(rffgt_0,139,42,102).  visible(rffgt_0,102). happensAt(walking(rffgt_0),102). orientation(rffgt_1,147,102). coords(rffgt_1,152,108,102).  visible(rffgt_1,102). happensAt(active(rffgt_1),102).
orientation(rffgt_0,135,103). coords(rffgt_0,139,42,103).  visible(rffgt_0,103). happensAt(walking(rffgt_0),103). orientation(rffgt_1,145,103). coords(rffgt_1,152,108,103).  visible(rffgt_1,103). happensAt(active(rffgt_1),103).
orientation(rffgt_0,135,104). coords(rffgt_0,139,42,104).  visible(rffgt_0,104). happensAt(walking(rffgt_0),104). orientation(rffgt_1,145,104). coords(rffgt_1,152,108,104).  visible(rffgt_1,104). happensAt(active(rffgt_1),104).
orientation(rffgt_0,135,105). coords(rffgt_0,140,42,105).  visible(rffgt_0,105). happensAt(walking(rffgt_0),105). orientation(rffgt_1,146,105). coords(rffgt_1,151,108,105).  visible(rffgt_1,105). happensAt(active(rffgt_1),105).
orientation(rffgt_0,135,106). coords(rffgt_0,140,42,106).  visible(rffgt_0,106). happensAt(walking(rffgt_0),106). orientation(rffgt_1,146,106). coords(rffgt_1,151,108,106).  visible(rffgt_1,106). happensAt(active(rffgt_1),106).
orientation(rffgt_0,135,107). coords(rffgt_0,141,42,107).  visible(rffgt_0,107). happensAt(walking(rffgt_0),107). orientation(rffgt_1,146,107). coords(rffgt_1,152,109,107).  visible(rffgt_1,107). happensAt(inactive(rffgt_1),107).
orientation(rffgt_0,135,108). coords(rffgt_0,142,42,108).  visible(rffgt_0,108). happensAt(walking(rffgt_0),108). orientation(rffgt_1,146,108). coords(rffgt_1,152,109,108).  visible(rffgt_1,108). happensAt(inactive(rffgt_1),108).
#show holdsAt/2.