
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

	orientation(wk2gt_2,141,70). coords(wk2gt_2,98,63,70).  visible(wk2gt_2,70). happensAt(active(wk2gt_2),70). orientation(wk2gt_3,107,70). coords(wk2gt_3,270,286,70).  happensAt(appear(wk2gt_3),70). happensAt(walking(wk2gt_3),70).
orientation(wk2gt_2,141,71). coords(wk2gt_2,98,63,71).  visible(wk2gt_2,71). happensAt(active(wk2gt_2),71). orientation(wk2gt_3,107,71). coords(wk2gt_3,263,284,71).  visible(wk2gt_3,71). happensAt(walking(wk2gt_3),71).
orientation(wk2gt_2,141,72). coords(wk2gt_2,98,63,72).  visible(wk2gt_2,72). happensAt(active(wk2gt_2),72). orientation(wk2gt_3,107,72). coords(wk2gt_3,258,282,72).  visible(wk2gt_3,72). happensAt(walking(wk2gt_3),72).
orientation(wk2gt_2,141,73). coords(wk2gt_2,98,63,73).  visible(wk2gt_2,73). happensAt(active(wk2gt_2),73). orientation(wk2gt_3,107,73). coords(wk2gt_3,256,280,73).  visible(wk2gt_3,73). happensAt(walking(wk2gt_3),73).
orientation(wk2gt_2,141,74). coords(wk2gt_2,98,63,74).  visible(wk2gt_2,74). happensAt(active(wk2gt_2),74). orientation(wk2gt_3,99,74). coords(wk2gt_3,255,280,74).  visible(wk2gt_3,74). happensAt(walking(wk2gt_3),74).
orientation(wk2gt_2,141,75). coords(wk2gt_2,98,63,75).  visible(wk2gt_2,75). happensAt(active(wk2gt_2),75). orientation(wk2gt_3,82,75). coords(wk2gt_3,254,279,75).  visible(wk2gt_3,75). happensAt(walking(wk2gt_3),75).
orientation(wk2gt_2,141,76). coords(wk2gt_2,98,63,76).  visible(wk2gt_2,76). happensAt(active(wk2gt_2),76). orientation(wk2gt_3,58,76). coords(wk2gt_3,262,278,76).  visible(wk2gt_3,76). happensAt(walking(wk2gt_3),76).
orientation(wk2gt_2,141,77). coords(wk2gt_2,98,63,77).  visible(wk2gt_2,77). happensAt(active(wk2gt_2),77). orientation(wk2gt_3,47,77). coords(wk2gt_3,261,278,77).  visible(wk2gt_3,77). happensAt(walking(wk2gt_3),77).
orientation(wk2gt_2,141,78). coords(wk2gt_2,98,63,78).  visible(wk2gt_2,78). happensAt(active(wk2gt_2),78). orientation(wk2gt_3,33,78). coords(wk2gt_3,259,278,78).  visible(wk2gt_3,78). happensAt(walking(wk2gt_3),78).
orientation(wk2gt_2,141,79). coords(wk2gt_2,98,63,79).  visible(wk2gt_2,79). happensAt(active(wk2gt_2),79). orientation(wk2gt_3,26,79). coords(wk2gt_3,256,278,79).  visible(wk2gt_3,79). happensAt(walking(wk2gt_3),79).
orientation(wk2gt_2,141,80). coords(wk2gt_2,98,63,80).  visible(wk2gt_2,80). happensAt(active(wk2gt_2),80). orientation(wk2gt_3,34,80). coords(wk2gt_3,247,277,80).  visible(wk2gt_3,80). happensAt(walking(wk2gt_3),80).
orientation(wk2gt_2,141,81). coords(wk2gt_2,98,63,81).  visible(wk2gt_2,81). happensAt(active(wk2gt_2),81). orientation(wk2gt_3,34,81). coords(wk2gt_3,242,276,81).  visible(wk2gt_3,81). happensAt(walking(wk2gt_3),81).
orientation(wk2gt_2,141,82). coords(wk2gt_2,98,63,82).  visible(wk2gt_2,82). happensAt(active(wk2gt_2),82). orientation(wk2gt_3,34,82). coords(wk2gt_3,239,274,82).  visible(wk2gt_3,82). happensAt(walking(wk2gt_3),82).
orientation(wk2gt_2,141,83). coords(wk2gt_2,98,63,83).  visible(wk2gt_2,83). happensAt(active(wk2gt_2),83). orientation(wk2gt_3,34,83). coords(wk2gt_3,238,271,83).  visible(wk2gt_3,83). happensAt(walking(wk2gt_3),83).
orientation(wk2gt_2,141,84). coords(wk2gt_2,98,63,84).  visible(wk2gt_2,84). happensAt(active(wk2gt_2),84). orientation(wk2gt_3,31,84). coords(wk2gt_3,237,269,84).  visible(wk2gt_3,84). happensAt(walking(wk2gt_3),84).
orientation(wk2gt_2,141,85). coords(wk2gt_2,98,63,85).  visible(wk2gt_2,85). happensAt(active(wk2gt_2),85). orientation(wk2gt_3,31,85). coords(wk2gt_3,236,267,85).  visible(wk2gt_3,85). happensAt(walking(wk2gt_3),85).
orientation(wk2gt_2,141,86). coords(wk2gt_2,98,63,86).  visible(wk2gt_2,86). happensAt(active(wk2gt_2),86). orientation(wk2gt_3,31,86). coords(wk2gt_3,235,264,86).  visible(wk2gt_3,86). happensAt(walking(wk2gt_3),86).
orientation(wk2gt_2,141,87). coords(wk2gt_2,98,63,87).  visible(wk2gt_2,87). happensAt(active(wk2gt_2),87). orientation(wk2gt_3,28,87). coords(wk2gt_3,235,262,87).  visible(wk2gt_3,87). happensAt(walking(wk2gt_3),87).
orientation(wk2gt_2,141,88). coords(wk2gt_2,98,63,88).  visible(wk2gt_2,88). happensAt(active(wk2gt_2),88). orientation(wk2gt_3,28,88). coords(wk2gt_3,233,260,88).  visible(wk2gt_3,88). happensAt(walking(wk2gt_3),88).
orientation(wk2gt_2,141,89). coords(wk2gt_2,98,63,89).  visible(wk2gt_2,89). happensAt(active(wk2gt_2),89). orientation(wk2gt_3,28,89). coords(wk2gt_3,232,259,89).  visible(wk2gt_3,89). happensAt(walking(wk2gt_3),89).
orientation(wk2gt_2,141,90). coords(wk2gt_2,98,63,90).  visible(wk2gt_2,90). happensAt(active(wk2gt_2),90). orientation(wk2gt_3,28,90). coords(wk2gt_3,230,259,90).  visible(wk2gt_3,90). happensAt(walking(wk2gt_3),90).
orientation(wk2gt_2,141,91). coords(wk2gt_2,98,62,91).  visible(wk2gt_2,91). happensAt(active(wk2gt_2),91). orientation(wk2gt_3,28,91). coords(wk2gt_3,228,258,91).  visible(wk2gt_3,91). happensAt(walking(wk2gt_3),91).
orientation(wk2gt_2,141,92). coords(wk2gt_2,98,62,92).  visible(wk2gt_2,92). happensAt(active(wk2gt_2),92). orientation(wk2gt_3,6,92). coords(wk2gt_3,225,256,92).  visible(wk2gt_3,92). happensAt(walking(wk2gt_3),92).
orientation(wk2gt_2,141,93). coords(wk2gt_2,98,62,93).  visible(wk2gt_2,93). happensAt(inactive(wk2gt_2),93). orientation(wk2gt_3,6,93). coords(wk2gt_3,222,254,93).  visible(wk2gt_3,93). happensAt(walking(wk2gt_3),93).
orientation(wk2gt_2,141,94). coords(wk2gt_2,98,62,94).  visible(wk2gt_2,94). happensAt(inactive(wk2gt_2),94). orientation(wk2gt_3,6,94). coords(wk2gt_3,219,252,94).  visible(wk2gt_3,94). happensAt(walking(wk2gt_3),94).
#show holdsAt/2.