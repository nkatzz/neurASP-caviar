
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

	orientation(rsfgt_0,160,84). coords(rsfgt_0,61,144,84).  visible(rsfgt_0,84). happensAt(walking(rsfgt_0),84). orientation(rsfgt_1,150,84). coords(rsfgt_1,166,112,84).  visible(rsfgt_1,84). happensAt(walking(rsfgt_1),84).
orientation(rsfgt_0,160,85). coords(rsfgt_0,61,144,85).  visible(rsfgt_0,85). happensAt(walking(rsfgt_0),85). orientation(rsfgt_1,150,85). coords(rsfgt_1,164,112,85).  visible(rsfgt_1,85). happensAt(walking(rsfgt_1),85).
orientation(rsfgt_0,160,86). coords(rsfgt_0,60,144,86).  visible(rsfgt_0,86). happensAt(walking(rsfgt_0),86). orientation(rsfgt_1,150,86). coords(rsfgt_1,162,112,86).  visible(rsfgt_1,86). happensAt(walking(rsfgt_1),86).
orientation(rsfgt_0,160,87). coords(rsfgt_0,60,146,87).  visible(rsfgt_0,87). happensAt(walking(rsfgt_0),87). orientation(rsfgt_1,150,87). coords(rsfgt_1,161,110,87).  visible(rsfgt_1,87). happensAt(walking(rsfgt_1),87).
orientation(rsfgt_0,160,88). coords(rsfgt_0,60,146,88).  visible(rsfgt_0,88). happensAt(walking(rsfgt_0),88). orientation(rsfgt_1,150,88). coords(rsfgt_1,161,110,88).  visible(rsfgt_1,88). happensAt(walking(rsfgt_1),88).
orientation(rsfgt_0,160,89). coords(rsfgt_0,58,146,89).  visible(rsfgt_0,89). happensAt(walking(rsfgt_0),89). orientation(rsfgt_1,150,89). coords(rsfgt_1,161,108,89).  visible(rsfgt_1,89). happensAt(walking(rsfgt_1),89).
orientation(rsfgt_0,160,90). coords(rsfgt_0,58,146,90).  visible(rsfgt_0,90). happensAt(walking(rsfgt_0),90). orientation(rsfgt_1,150,90). coords(rsfgt_1,161,107,90).  visible(rsfgt_1,90). happensAt(walking(rsfgt_1),90).
orientation(rsfgt_0,160,91). coords(rsfgt_0,58,147,91).  visible(rsfgt_0,91). happensAt(walking(rsfgt_0),91). orientation(rsfgt_1,150,91). coords(rsfgt_1,161,106,91).  visible(rsfgt_1,91). happensAt(active(rsfgt_1),91).
orientation(rsfgt_0,160,92). coords(rsfgt_0,57,147,92).  visible(rsfgt_0,92). happensAt(walking(rsfgt_0),92). orientation(rsfgt_1,150,92). coords(rsfgt_1,161,106,92).  visible(rsfgt_1,92). happensAt(active(rsfgt_1),92).
orientation(rsfgt_0,160,93). coords(rsfgt_0,56,148,93).  visible(rsfgt_0,93). happensAt(walking(rsfgt_0),93). orientation(rsfgt_1,150,93). coords(rsfgt_1,161,106,93).  visible(rsfgt_1,93). happensAt(active(rsfgt_1),93).
orientation(rsfgt_0,160,94). coords(rsfgt_0,56,148,94).  visible(rsfgt_0,94). happensAt(walking(rsfgt_0),94). orientation(rsfgt_1,150,94). coords(rsfgt_1,165,104,94).  visible(rsfgt_1,94). happensAt(active(rsfgt_1),94).
orientation(rsfgt_0,160,95). coords(rsfgt_0,56,149,95).  visible(rsfgt_0,95). happensAt(walking(rsfgt_0),95). orientation(rsfgt_1,150,95). coords(rsfgt_1,165,104,95).  visible(rsfgt_1,95). happensAt(active(rsfgt_1),95).
orientation(rsfgt_0,160,96). coords(rsfgt_0,56,150,96).  visible(rsfgt_0,96). happensAt(walking(rsfgt_0),96). orientation(rsfgt_1,150,96). coords(rsfgt_1,165,103,96).  visible(rsfgt_1,96). happensAt(active(rsfgt_1),96).
orientation(rsfgt_0,160,97). coords(rsfgt_0,56,151,97).  visible(rsfgt_0,97). happensAt(walking(rsfgt_0),97). orientation(rsfgt_1,150,97). coords(rsfgt_1,165,103,97).  visible(rsfgt_1,97). happensAt(active(rsfgt_1),97).
orientation(rsfgt_0,160,98). coords(rsfgt_0,55,151,98).  visible(rsfgt_0,98). happensAt(walking(rsfgt_0),98). orientation(rsfgt_1,150,98). coords(rsfgt_1,165,103,98).  visible(rsfgt_1,98). happensAt(active(rsfgt_1),98).
orientation(rsfgt_0,160,99). coords(rsfgt_0,55,151,99).  visible(rsfgt_0,99). happensAt(walking(rsfgt_0),99). orientation(rsfgt_1,150,99). coords(rsfgt_1,165,103,99).  visible(rsfgt_1,99). happensAt(active(rsfgt_1),99).
orientation(rsfgt_0,160,100). coords(rsfgt_0,54,151,100).  visible(rsfgt_0,100). happensAt(walking(rsfgt_0),100). orientation(rsfgt_1,150,100). coords(rsfgt_1,165,103,100).  visible(rsfgt_1,100). happensAt(active(rsfgt_1),100).
orientation(rsfgt_0,160,101). coords(rsfgt_0,53,151,101).  visible(rsfgt_0,101). happensAt(walking(rsfgt_0),101). orientation(rsfgt_1,150,101). coords(rsfgt_1,165,103,101).  visible(rsfgt_1,101). happensAt(active(rsfgt_1),101).
orientation(rsfgt_0,160,102). coords(rsfgt_0,53,152,102).  visible(rsfgt_0,102). happensAt(walking(rsfgt_0),102). orientation(rsfgt_1,150,102). coords(rsfgt_1,165,103,102).  visible(rsfgt_1,102). happensAt(active(rsfgt_1),102).
orientation(rsfgt_0,160,103). coords(rsfgt_0,52,152,103).  visible(rsfgt_0,103). happensAt(walking(rsfgt_0),103). orientation(rsfgt_1,150,103). coords(rsfgt_1,165,104,103).  visible(rsfgt_1,103). happensAt(active(rsfgt_1),103).
orientation(rsfgt_0,160,104). coords(rsfgt_0,49,152,104).  visible(rsfgt_0,104). happensAt(walking(rsfgt_0),104). orientation(rsfgt_1,150,104). coords(rsfgt_1,165,104,104).  visible(rsfgt_1,104). happensAt(active(rsfgt_1),104).
orientation(rsfgt_0,160,105). coords(rsfgt_0,49,152,105).  visible(rsfgt_0,105). happensAt(walking(rsfgt_0),105). orientation(rsfgt_1,150,105). coords(rsfgt_1,165,104,105).  visible(rsfgt_1,105). happensAt(active(rsfgt_1),105).
orientation(rsfgt_0,160,106). coords(rsfgt_0,49,153,106).  visible(rsfgt_0,106). happensAt(walking(rsfgt_0),106). orientation(rsfgt_1,150,106). coords(rsfgt_1,166,106,106).  visible(rsfgt_1,106). happensAt(active(rsfgt_1),106).
orientation(rsfgt_0,160,107). coords(rsfgt_0,49,153,107).  visible(rsfgt_0,107). happensAt(walking(rsfgt_0),107). orientation(rsfgt_1,150,107). coords(rsfgt_1,166,106,107).  visible(rsfgt_1,107). happensAt(active(rsfgt_1),107).
orientation(rsfgt_0,160,108). coords(rsfgt_0,49,153,108).  visible(rsfgt_0,108). happensAt(walking(rsfgt_0),108). orientation(rsfgt_1,150,108). coords(rsfgt_1,166,106,108).  visible(rsfgt_1,108). happensAt(active(rsfgt_1),108).
#show holdsAt/2.