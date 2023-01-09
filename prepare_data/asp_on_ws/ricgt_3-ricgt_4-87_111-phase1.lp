
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

	orientation(ricgt_3,2,87). coords(ricgt_3,231,219,87).  visible(ricgt_3,87). happensAt(walking(ricgt_3),87). orientation(ricgt_4,172,87). coords(ricgt_4,27,133,87).  visible(ricgt_4,87). happensAt(walking(ricgt_4),87).
orientation(ricgt_3,172,88). coords(ricgt_3,229,218,88).  visible(ricgt_3,88). happensAt(walking(ricgt_3),88). orientation(ricgt_4,172,88). coords(ricgt_4,27,133,88).  visible(ricgt_4,88). happensAt(walking(ricgt_4),88).
orientation(ricgt_3,168,89). coords(ricgt_3,226,216,89).  visible(ricgt_3,89). happensAt(walking(ricgt_3),89). orientation(ricgt_4,172,89). coords(ricgt_4,27,133,89).  visible(ricgt_4,89). happensAt(walking(ricgt_4),89).
orientation(ricgt_3,161,90). coords(ricgt_3,221,214,90).  visible(ricgt_3,90). happensAt(walking(ricgt_3),90). orientation(ricgt_4,172,90). coords(ricgt_4,27,133,90).  visible(ricgt_4,90). happensAt(walking(ricgt_4),90).
orientation(ricgt_3,158,91). coords(ricgt_3,219,213,91).  visible(ricgt_3,91). happensAt(walking(ricgt_3),91). orientation(ricgt_4,172,91). coords(ricgt_4,27,133,91).  visible(ricgt_4,91). happensAt(walking(ricgt_4),91).
orientation(ricgt_3,157,92). coords(ricgt_3,218,212,92).  visible(ricgt_3,92). happensAt(walking(ricgt_3),92). orientation(ricgt_4,172,92). coords(ricgt_4,27,133,92).  visible(ricgt_4,92). happensAt(walking(ricgt_4),92).
orientation(ricgt_3,157,93). coords(ricgt_3,217,210,93).  visible(ricgt_3,93). happensAt(walking(ricgt_3),93). orientation(ricgt_4,172,93). coords(ricgt_4,26,133,93).  visible(ricgt_4,93). happensAt(walking(ricgt_4),93).
orientation(ricgt_3,154,94). coords(ricgt_3,216,208,94).  visible(ricgt_3,94). happensAt(walking(ricgt_3),94). orientation(ricgt_4,172,94). coords(ricgt_4,26,134,94).  visible(ricgt_4,94). happensAt(walking(ricgt_4),94).
orientation(ricgt_3,153,95). coords(ricgt_3,215,205,95).  visible(ricgt_3,95). happensAt(walking(ricgt_3),95). orientation(ricgt_4,172,95). coords(ricgt_4,26,134,95).  visible(ricgt_4,95). happensAt(walking(ricgt_4),95).
orientation(ricgt_3,153,96). coords(ricgt_3,213,203,96).  visible(ricgt_3,96). happensAt(walking(ricgt_3),96). orientation(ricgt_4,172,96). coords(ricgt_4,26,134,96).  visible(ricgt_4,96). happensAt(walking(ricgt_4),96).
orientation(ricgt_3,152,97). coords(ricgt_3,212,199,97).  visible(ricgt_3,97). happensAt(walking(ricgt_3),97). orientation(ricgt_4,172,97). coords(ricgt_4,26,134,97).  visible(ricgt_4,97). happensAt(walking(ricgt_4),97).
orientation(ricgt_3,150,98). coords(ricgt_3,210,196,98).  visible(ricgt_3,98). happensAt(walking(ricgt_3),98). orientation(ricgt_4,172,98). coords(ricgt_4,26,134,98).  visible(ricgt_4,98). happensAt(walking(ricgt_4),98).
orientation(ricgt_3,155,99). coords(ricgt_3,212,190,99).  visible(ricgt_3,99). happensAt(walking(ricgt_3),99). orientation(ricgt_4,172,99). coords(ricgt_4,26,134,99).  visible(ricgt_4,99). happensAt(walking(ricgt_4),99).
orientation(ricgt_3,158,100). coords(ricgt_3,212,188,100).  visible(ricgt_3,100). happensAt(walking(ricgt_3),100). orientation(ricgt_4,172,100). coords(ricgt_4,26,134,100).  visible(ricgt_4,100). happensAt(walking(ricgt_4),100).
orientation(ricgt_3,159,101). coords(ricgt_3,209,186,101).  visible(ricgt_3,101). happensAt(walking(ricgt_3),101). orientation(ricgt_4,172,101). coords(ricgt_4,26,134,101).  visible(ricgt_4,101). happensAt(walking(ricgt_4),101).
orientation(ricgt_3,161,102). coords(ricgt_3,208,184,102).  visible(ricgt_3,102). happensAt(walking(ricgt_3),102). orientation(ricgt_4,172,102). coords(ricgt_4,26,134,102).  visible(ricgt_4,102). happensAt(walking(ricgt_4),102).
orientation(ricgt_3,163,103). coords(ricgt_3,206,181,103).  visible(ricgt_3,103). happensAt(walking(ricgt_3),103). orientation(ricgt_4,172,103). coords(ricgt_4,26,134,103).  visible(ricgt_4,103). happensAt(walking(ricgt_4),103).
orientation(ricgt_3,161,104). coords(ricgt_3,202,179,104).  visible(ricgt_3,104). happensAt(walking(ricgt_3),104). orientation(ricgt_4,172,104). coords(ricgt_4,26,134,104).  visible(ricgt_4,104). happensAt(walking(ricgt_4),104).
orientation(ricgt_3,159,105). coords(ricgt_3,201,176,105).  visible(ricgt_3,105). happensAt(walking(ricgt_3),105). orientation(ricgt_4,172,105). coords(ricgt_4,26,134,105).  visible(ricgt_4,105). happensAt(walking(ricgt_4),105).
orientation(ricgt_3,158,106). coords(ricgt_3,201,175,106).  visible(ricgt_3,106). happensAt(walking(ricgt_3),106). orientation(ricgt_4,172,106). coords(ricgt_4,26,134,106).  visible(ricgt_4,106). happensAt(walking(ricgt_4),106).
orientation(ricgt_3,157,107). coords(ricgt_3,198,174,107).  visible(ricgt_3,107). happensAt(walking(ricgt_3),107). orientation(ricgt_4,172,107). coords(ricgt_4,26,134,107).  visible(ricgt_4,107). happensAt(walking(ricgt_4),107).
orientation(ricgt_3,155,108). coords(ricgt_3,197,173,108).  visible(ricgt_3,108). happensAt(walking(ricgt_3),108). orientation(ricgt_4,172,108). coords(ricgt_4,26,134,108).  visible(ricgt_4,108). happensAt(walking(ricgt_4),108).
orientation(ricgt_3,155,109). coords(ricgt_3,196,172,109).  visible(ricgt_3,109). happensAt(walking(ricgt_3),109). orientation(ricgt_4,172,109). coords(ricgt_4,26,134,109).  visible(ricgt_4,109). happensAt(walking(ricgt_4),109).
orientation(ricgt_3,155,110). coords(ricgt_3,194,171,110).  visible(ricgt_3,110). happensAt(walking(ricgt_3),110). orientation(ricgt_4,172,110). coords(ricgt_4,26,134,110).  visible(ricgt_4,110). happensAt(walking(ricgt_4),110).
orientation(ricgt_3,153,111). coords(ricgt_3,192,170,111).  visible(ricgt_3,111). happensAt(walking(ricgt_3),111). orientation(ricgt_4,172,111). coords(ricgt_4,26,134,111).  visible(ricgt_4,111). happensAt(walking(ricgt_4),111).
#show holdsAt/2.