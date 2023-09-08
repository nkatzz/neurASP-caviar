
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

	orientation(wk2gt_1,134,82). coords(wk2gt_1,112,53,82).  visible(wk2gt_1,82). happensAt(inactive(wk2gt_1),82). orientation(wk2gt_3,34,82). coords(wk2gt_3,239,274,82).  visible(wk2gt_3,82). happensAt(walking(wk2gt_3),82).
orientation(wk2gt_1,139,83). coords(wk2gt_1,112,53,83).  visible(wk2gt_1,83). happensAt(inactive(wk2gt_1),83). orientation(wk2gt_3,34,83). coords(wk2gt_3,238,271,83).  visible(wk2gt_3,83). happensAt(walking(wk2gt_3),83).
orientation(wk2gt_1,137,84). coords(wk2gt_1,113,53,84).  visible(wk2gt_1,84). happensAt(inactive(wk2gt_1),84). orientation(wk2gt_3,31,84). coords(wk2gt_3,237,269,84).  visible(wk2gt_3,84). happensAt(walking(wk2gt_3),84).
orientation(wk2gt_1,135,85). coords(wk2gt_1,114,53,85).  visible(wk2gt_1,85). happensAt(inactive(wk2gt_1),85). orientation(wk2gt_3,31,85). coords(wk2gt_3,236,267,85).  visible(wk2gt_3,85). happensAt(walking(wk2gt_3),85).
orientation(wk2gt_1,136,86). coords(wk2gt_1,114,53,86).  visible(wk2gt_1,86). happensAt(inactive(wk2gt_1),86). orientation(wk2gt_3,31,86). coords(wk2gt_3,235,264,86).  visible(wk2gt_3,86). happensAt(walking(wk2gt_3),86).
orientation(wk2gt_1,136,87). coords(wk2gt_1,114,53,87).  visible(wk2gt_1,87). happensAt(inactive(wk2gt_1),87). orientation(wk2gt_3,28,87). coords(wk2gt_3,235,262,87).  visible(wk2gt_3,87). happensAt(walking(wk2gt_3),87).
orientation(wk2gt_1,136,88). coords(wk2gt_1,114,53,88).  visible(wk2gt_1,88). happensAt(inactive(wk2gt_1),88). orientation(wk2gt_3,28,88). coords(wk2gt_3,233,260,88).  visible(wk2gt_3,88). happensAt(walking(wk2gt_3),88).
orientation(wk2gt_1,134,89). coords(wk2gt_1,114,53,89).  visible(wk2gt_1,89). happensAt(inactive(wk2gt_1),89). orientation(wk2gt_3,28,89). coords(wk2gt_3,232,259,89).  visible(wk2gt_3,89). happensAt(walking(wk2gt_3),89).
orientation(wk2gt_1,138,90). coords(wk2gt_1,114,53,90).  visible(wk2gt_1,90). happensAt(inactive(wk2gt_1),90). orientation(wk2gt_3,28,90). coords(wk2gt_3,230,259,90).  visible(wk2gt_3,90). happensAt(walking(wk2gt_3),90).
orientation(wk2gt_1,133,91). coords(wk2gt_1,115,53,91).  visible(wk2gt_1,91). happensAt(inactive(wk2gt_1),91). orientation(wk2gt_3,28,91). coords(wk2gt_3,228,258,91).  visible(wk2gt_3,91). happensAt(walking(wk2gt_3),91).
orientation(wk2gt_1,135,92). coords(wk2gt_1,115,53,92).  visible(wk2gt_1,92). happensAt(inactive(wk2gt_1),92). orientation(wk2gt_3,6,92). coords(wk2gt_3,225,256,92).  visible(wk2gt_3,92). happensAt(walking(wk2gt_3),92).
orientation(wk2gt_1,136,93). coords(wk2gt_1,115,53,93).  visible(wk2gt_1,93). happensAt(inactive(wk2gt_1),93). orientation(wk2gt_3,6,93). coords(wk2gt_3,222,254,93).  visible(wk2gt_3,93). happensAt(walking(wk2gt_3),93).
orientation(wk2gt_1,137,94). coords(wk2gt_1,115,53,94).  visible(wk2gt_1,94). happensAt(inactive(wk2gt_1),94). orientation(wk2gt_3,6,94). coords(wk2gt_3,219,252,94).  visible(wk2gt_3,94). happensAt(walking(wk2gt_3),94).
orientation(wk2gt_1,138,95). coords(wk2gt_1,115,53,95).  visible(wk2gt_1,95). happensAt(inactive(wk2gt_1),95). orientation(wk2gt_3,6,95). coords(wk2gt_3,218,250,95).  visible(wk2gt_3,95). happensAt(walking(wk2gt_3),95).
orientation(wk2gt_1,138,96). coords(wk2gt_1,115,53,96).  visible(wk2gt_1,96). happensAt(inactive(wk2gt_1),96). orientation(wk2gt_3,6,96). coords(wk2gt_3,217,247,96).  visible(wk2gt_3,96). happensAt(walking(wk2gt_3),96).
orientation(wk2gt_1,139,97). coords(wk2gt_1,115,53,97).  visible(wk2gt_1,97). happensAt(inactive(wk2gt_1),97). orientation(wk2gt_3,6,97). coords(wk2gt_3,217,244,97).  visible(wk2gt_3,97). happensAt(walking(wk2gt_3),97).
orientation(wk2gt_1,136,98). coords(wk2gt_1,115,53,98).  visible(wk2gt_1,98). happensAt(inactive(wk2gt_1),98). orientation(wk2gt_3,176,98). coords(wk2gt_3,216,241,98).  visible(wk2gt_3,98). happensAt(walking(wk2gt_3),98).
orientation(wk2gt_1,136,99). coords(wk2gt_1,115,52,99).  visible(wk2gt_1,99). happensAt(inactive(wk2gt_1),99). orientation(wk2gt_3,176,99). coords(wk2gt_3,215,239,99).  visible(wk2gt_3,99). happensAt(walking(wk2gt_3),99).
orientation(wk2gt_1,135,100). coords(wk2gt_1,115,52,100).  visible(wk2gt_1,100). happensAt(inactive(wk2gt_1),100). orientation(wk2gt_3,167,100). coords(wk2gt_3,214,236,100).  visible(wk2gt_3,100). happensAt(walking(wk2gt_3),100).
orientation(wk2gt_1,135,101). coords(wk2gt_1,115,52,101).  visible(wk2gt_1,101). happensAt(inactive(wk2gt_1),101). orientation(wk2gt_3,167,101). coords(wk2gt_3,213,234,101).  visible(wk2gt_3,101). happensAt(walking(wk2gt_3),101).
orientation(wk2gt_1,138,102). coords(wk2gt_1,115,52,102).  visible(wk2gt_1,102). happensAt(inactive(wk2gt_1),102). orientation(wk2gt_3,167,102). coords(wk2gt_3,213,233,102).  visible(wk2gt_3,102). happensAt(walking(wk2gt_3),102).
orientation(wk2gt_1,139,103). coords(wk2gt_1,115,52,103).  visible(wk2gt_1,103). happensAt(inactive(wk2gt_1),103). orientation(wk2gt_3,167,103). coords(wk2gt_3,211,232,103).  visible(wk2gt_3,103). happensAt(walking(wk2gt_3),103).
orientation(wk2gt_1,135,104). coords(wk2gt_1,115,52,104).  visible(wk2gt_1,104). happensAt(inactive(wk2gt_1),104). orientation(wk2gt_3,167,104). coords(wk2gt_3,210,231,104).  visible(wk2gt_3,104). happensAt(walking(wk2gt_3),104).
orientation(wk2gt_1,135,105). coords(wk2gt_1,115,52,105).  visible(wk2gt_1,105). happensAt(inactive(wk2gt_1),105). orientation(wk2gt_3,167,105). coords(wk2gt_3,209,228,105).  visible(wk2gt_3,105). happensAt(walking(wk2gt_3),105).
orientation(wk2gt_1,134,106). coords(wk2gt_1,115,52,106).  visible(wk2gt_1,106). happensAt(inactive(wk2gt_1),106). orientation(wk2gt_3,167,106). coords(wk2gt_3,207,226,106).  visible(wk2gt_3,106). happensAt(walking(wk2gt_3),106).
#show holdsAt/2.