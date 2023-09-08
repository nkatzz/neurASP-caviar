
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

	orientation(wk2gt_1,137,84). coords(wk2gt_1,113,53,84).  visible(wk2gt_1,84). happensAt(inactive(wk2gt_1),84). orientation(wk2gt_2,141,84). coords(wk2gt_2,98,63,84).  visible(wk2gt_2,84). happensAt(active(wk2gt_2),84).
orientation(wk2gt_1,135,85). coords(wk2gt_1,114,53,85).  visible(wk2gt_1,85). happensAt(inactive(wk2gt_1),85). orientation(wk2gt_2,141,85). coords(wk2gt_2,98,63,85).  visible(wk2gt_2,85). happensAt(active(wk2gt_2),85).
orientation(wk2gt_1,136,86). coords(wk2gt_1,114,53,86).  visible(wk2gt_1,86). happensAt(inactive(wk2gt_1),86). orientation(wk2gt_2,141,86). coords(wk2gt_2,98,63,86).  visible(wk2gt_2,86). happensAt(active(wk2gt_2),86).
orientation(wk2gt_1,136,87). coords(wk2gt_1,114,53,87).  visible(wk2gt_1,87). happensAt(inactive(wk2gt_1),87). orientation(wk2gt_2,141,87). coords(wk2gt_2,98,63,87).  visible(wk2gt_2,87). happensAt(active(wk2gt_2),87).
orientation(wk2gt_1,136,88). coords(wk2gt_1,114,53,88).  visible(wk2gt_1,88). happensAt(inactive(wk2gt_1),88). orientation(wk2gt_2,141,88). coords(wk2gt_2,98,63,88).  visible(wk2gt_2,88). happensAt(active(wk2gt_2),88).
orientation(wk2gt_1,134,89). coords(wk2gt_1,114,53,89).  visible(wk2gt_1,89). happensAt(inactive(wk2gt_1),89). orientation(wk2gt_2,141,89). coords(wk2gt_2,98,63,89).  visible(wk2gt_2,89). happensAt(active(wk2gt_2),89).
orientation(wk2gt_1,138,90). coords(wk2gt_1,114,53,90).  visible(wk2gt_1,90). happensAt(inactive(wk2gt_1),90). orientation(wk2gt_2,141,90). coords(wk2gt_2,98,63,90).  visible(wk2gt_2,90). happensAt(active(wk2gt_2),90).
orientation(wk2gt_1,133,91). coords(wk2gt_1,115,53,91).  visible(wk2gt_1,91). happensAt(inactive(wk2gt_1),91). orientation(wk2gt_2,141,91). coords(wk2gt_2,98,62,91).  visible(wk2gt_2,91). happensAt(active(wk2gt_2),91).
orientation(wk2gt_1,135,92). coords(wk2gt_1,115,53,92).  visible(wk2gt_1,92). happensAt(inactive(wk2gt_1),92). orientation(wk2gt_2,141,92). coords(wk2gt_2,98,62,92).  visible(wk2gt_2,92). happensAt(active(wk2gt_2),92).
orientation(wk2gt_1,136,93). coords(wk2gt_1,115,53,93).  visible(wk2gt_1,93). happensAt(inactive(wk2gt_1),93). orientation(wk2gt_2,141,93). coords(wk2gt_2,98,62,93).  visible(wk2gt_2,93). happensAt(inactive(wk2gt_2),93).
orientation(wk2gt_1,137,94). coords(wk2gt_1,115,53,94).  visible(wk2gt_1,94). happensAt(inactive(wk2gt_1),94). orientation(wk2gt_2,141,94). coords(wk2gt_2,98,62,94).  visible(wk2gt_2,94). happensAt(inactive(wk2gt_2),94).
orientation(wk2gt_1,138,95). coords(wk2gt_1,115,53,95).  visible(wk2gt_1,95). happensAt(inactive(wk2gt_1),95). orientation(wk2gt_2,141,95). coords(wk2gt_2,98,62,95).  visible(wk2gt_2,95). happensAt(inactive(wk2gt_2),95).
orientation(wk2gt_1,138,96). coords(wk2gt_1,115,53,96).  visible(wk2gt_1,96). happensAt(inactive(wk2gt_1),96). orientation(wk2gt_2,141,96). coords(wk2gt_2,98,62,96).  visible(wk2gt_2,96). happensAt(inactive(wk2gt_2),96).
orientation(wk2gt_1,139,97). coords(wk2gt_1,115,53,97).  visible(wk2gt_1,97). happensAt(inactive(wk2gt_1),97). orientation(wk2gt_2,141,97). coords(wk2gt_2,98,62,97).  visible(wk2gt_2,97). happensAt(inactive(wk2gt_2),97).
orientation(wk2gt_1,136,98). coords(wk2gt_1,115,53,98).  visible(wk2gt_1,98). happensAt(inactive(wk2gt_1),98). orientation(wk2gt_2,141,98). coords(wk2gt_2,98,62,98).  visible(wk2gt_2,98). happensAt(inactive(wk2gt_2),98).
orientation(wk2gt_1,136,99). coords(wk2gt_1,115,52,99).  visible(wk2gt_1,99). happensAt(inactive(wk2gt_1),99). orientation(wk2gt_2,141,99). coords(wk2gt_2,98,62,99).  visible(wk2gt_2,99). happensAt(inactive(wk2gt_2),99).
orientation(wk2gt_1,135,100). coords(wk2gt_1,115,52,100).  visible(wk2gt_1,100). happensAt(inactive(wk2gt_1),100). orientation(wk2gt_2,141,100). coords(wk2gt_2,98,62,100).  visible(wk2gt_2,100). happensAt(inactive(wk2gt_2),100).
orientation(wk2gt_1,135,101). coords(wk2gt_1,115,52,101).  visible(wk2gt_1,101). happensAt(inactive(wk2gt_1),101). orientation(wk2gt_2,141,101). coords(wk2gt_2,98,62,101).  visible(wk2gt_2,101). happensAt(inactive(wk2gt_2),101).
orientation(wk2gt_1,138,102). coords(wk2gt_1,115,52,102).  visible(wk2gt_1,102). happensAt(inactive(wk2gt_1),102). orientation(wk2gt_2,141,102). coords(wk2gt_2,98,62,102).  visible(wk2gt_2,102). happensAt(inactive(wk2gt_2),102).
orientation(wk2gt_1,139,103). coords(wk2gt_1,115,52,103).  visible(wk2gt_1,103). happensAt(inactive(wk2gt_1),103). orientation(wk2gt_2,141,103). coords(wk2gt_2,98,62,103).  visible(wk2gt_2,103). happensAt(inactive(wk2gt_2),103).
orientation(wk2gt_1,135,104). coords(wk2gt_1,115,52,104).  visible(wk2gt_1,104). happensAt(inactive(wk2gt_1),104). orientation(wk2gt_2,141,104). coords(wk2gt_2,98,62,104).  visible(wk2gt_2,104). happensAt(inactive(wk2gt_2),104).
orientation(wk2gt_1,135,105). coords(wk2gt_1,115,52,105).  visible(wk2gt_1,105). happensAt(inactive(wk2gt_1),105). orientation(wk2gt_2,141,105). coords(wk2gt_2,98,62,105).  visible(wk2gt_2,105). happensAt(inactive(wk2gt_2),105).
orientation(wk2gt_1,134,106). coords(wk2gt_1,115,52,106).  visible(wk2gt_1,106). happensAt(inactive(wk2gt_1),106). orientation(wk2gt_2,141,106). coords(wk2gt_2,98,62,106).  visible(wk2gt_2,106). happensAt(inactive(wk2gt_2),106).
orientation(wk2gt_1,136,107). coords(wk2gt_1,115,52,107).  visible(wk2gt_1,107). happensAt(inactive(wk2gt_1),107). orientation(wk2gt_2,141,107). coords(wk2gt_2,98,62,107).  visible(wk2gt_2,107). happensAt(inactive(wk2gt_2),107).
orientation(wk2gt_1,137,108). coords(wk2gt_1,115,52,108).  visible(wk2gt_1,108). happensAt(inactive(wk2gt_1),108). orientation(wk2gt_2,141,108). coords(wk2gt_2,98,62,108).  visible(wk2gt_2,108). happensAt(inactive(wk2gt_2),108).
#show holdsAt/2.