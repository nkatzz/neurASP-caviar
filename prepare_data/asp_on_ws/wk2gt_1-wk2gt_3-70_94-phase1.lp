
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

	orientation(wk2gt_1,135,70). coords(wk2gt_1,113,53,70).  visible(wk2gt_1,70). happensAt(active(wk2gt_1),70). orientation(wk2gt_3,107,70). coords(wk2gt_3,270,286,70).  happensAt(appear(wk2gt_3),70). happensAt(walking(wk2gt_3),70).
orientation(wk2gt_1,135,71). coords(wk2gt_1,114,53,71).  visible(wk2gt_1,71). happensAt(active(wk2gt_1),71). orientation(wk2gt_3,107,71). coords(wk2gt_3,263,284,71).  visible(wk2gt_3,71). happensAt(walking(wk2gt_3),71).
orientation(wk2gt_1,135,72). coords(wk2gt_1,114,53,72).  visible(wk2gt_1,72). happensAt(active(wk2gt_1),72). orientation(wk2gt_3,107,72). coords(wk2gt_3,258,282,72).  visible(wk2gt_3,72). happensAt(walking(wk2gt_3),72).
orientation(wk2gt_1,135,73). coords(wk2gt_1,114,53,73).  visible(wk2gt_1,73). happensAt(active(wk2gt_1),73). orientation(wk2gt_3,107,73). coords(wk2gt_3,256,280,73).  visible(wk2gt_3,73). happensAt(walking(wk2gt_3),73).
orientation(wk2gt_1,138,74). coords(wk2gt_1,114,53,74).  visible(wk2gt_1,74). happensAt(active(wk2gt_1),74). orientation(wk2gt_3,99,74). coords(wk2gt_3,255,280,74).  visible(wk2gt_3,74). happensAt(walking(wk2gt_3),74).
orientation(wk2gt_1,137,75). coords(wk2gt_1,114,53,75).  visible(wk2gt_1,75). happensAt(inactive(wk2gt_1),75). orientation(wk2gt_3,82,75). coords(wk2gt_3,254,279,75).  visible(wk2gt_3,75). happensAt(walking(wk2gt_3),75).
orientation(wk2gt_1,137,76). coords(wk2gt_1,115,53,76).  visible(wk2gt_1,76). happensAt(inactive(wk2gt_1),76). orientation(wk2gt_3,58,76). coords(wk2gt_3,262,278,76).  visible(wk2gt_3,76). happensAt(walking(wk2gt_3),76).
orientation(wk2gt_1,136,77). coords(wk2gt_1,115,53,77).  visible(wk2gt_1,77). happensAt(inactive(wk2gt_1),77). orientation(wk2gt_3,47,77). coords(wk2gt_3,261,278,77).  visible(wk2gt_3,77). happensAt(walking(wk2gt_3),77).
orientation(wk2gt_1,141,78). coords(wk2gt_1,114,53,78).  visible(wk2gt_1,78). happensAt(inactive(wk2gt_1),78). orientation(wk2gt_3,33,78). coords(wk2gt_3,259,278,78).  visible(wk2gt_3,78). happensAt(walking(wk2gt_3),78).
orientation(wk2gt_1,139,79). coords(wk2gt_1,114,53,79).  visible(wk2gt_1,79). happensAt(inactive(wk2gt_1),79). orientation(wk2gt_3,26,79). coords(wk2gt_3,256,278,79).  visible(wk2gt_3,79). happensAt(walking(wk2gt_3),79).
orientation(wk2gt_1,135,80). coords(wk2gt_1,113,53,80).  visible(wk2gt_1,80). happensAt(inactive(wk2gt_1),80). orientation(wk2gt_3,34,80). coords(wk2gt_3,247,277,80).  visible(wk2gt_3,80). happensAt(walking(wk2gt_3),80).
orientation(wk2gt_1,134,81). coords(wk2gt_1,113,53,81).  visible(wk2gt_1,81). happensAt(inactive(wk2gt_1),81). orientation(wk2gt_3,34,81). coords(wk2gt_3,242,276,81).  visible(wk2gt_3,81). happensAt(walking(wk2gt_3),81).
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
#show holdsAt/2.