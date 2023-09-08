
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

	orientation(lb2gt_0,141,72). coords(lb2gt_0,193,124,72).  visible(lb2gt_0,72). happensAt(active(lb2gt_0),72). orientation(lb2gt_3,146,72). coords(lb2gt_3,14,46,72).  visible(lb2gt_3,72). happensAt(walking(lb2gt_3),72).
orientation(lb2gt_0,141,73). coords(lb2gt_0,193,124,73).  visible(lb2gt_0,73). happensAt(active(lb2gt_0),73). orientation(lb2gt_3,146,73). coords(lb2gt_3,14,46,73).  visible(lb2gt_3,73). happensAt(walking(lb2gt_3),73).
orientation(lb2gt_0,141,74). coords(lb2gt_0,193,124,74).  visible(lb2gt_0,74). happensAt(active(lb2gt_0),74). orientation(lb2gt_3,146,74). coords(lb2gt_3,14,46,74).  visible(lb2gt_3,74). happensAt(walking(lb2gt_3),74).
orientation(lb2gt_0,141,75). coords(lb2gt_0,193,124,75).  visible(lb2gt_0,75). happensAt(active(lb2gt_0),75). orientation(lb2gt_3,146,75). coords(lb2gt_3,14,46,75).  visible(lb2gt_3,75). happensAt(walking(lb2gt_3),75).
orientation(lb2gt_0,141,76). coords(lb2gt_0,193,124,76).  visible(lb2gt_0,76). happensAt(active(lb2gt_0),76). orientation(lb2gt_3,146,76). coords(lb2gt_3,14,46,76).  visible(lb2gt_3,76). happensAt(walking(lb2gt_3),76).
orientation(lb2gt_0,141,77). coords(lb2gt_0,193,124,77).  visible(lb2gt_0,77). happensAt(active(lb2gt_0),77). orientation(lb2gt_3,146,77). coords(lb2gt_3,14,46,77).  visible(lb2gt_3,77). happensAt(walking(lb2gt_3),77).
orientation(lb2gt_0,141,78). coords(lb2gt_0,193,124,78).  visible(lb2gt_0,78). happensAt(active(lb2gt_0),78). orientation(lb2gt_3,146,78). coords(lb2gt_3,14,47,78).  visible(lb2gt_3,78). happensAt(walking(lb2gt_3),78).
orientation(lb2gt_0,141,79). coords(lb2gt_0,193,124,79).  visible(lb2gt_0,79). happensAt(active(lb2gt_0),79). orientation(lb2gt_3,146,79). coords(lb2gt_3,14,47,79).  visible(lb2gt_3,79). happensAt(walking(lb2gt_3),79).
orientation(lb2gt_0,141,80). coords(lb2gt_0,193,124,80).  visible(lb2gt_0,80). happensAt(active(lb2gt_0),80). orientation(lb2gt_3,146,80). coords(lb2gt_3,14,47,80).  visible(lb2gt_3,80). happensAt(walking(lb2gt_3),80).
orientation(lb2gt_0,141,81). coords(lb2gt_0,193,124,81).  visible(lb2gt_0,81). happensAt(active(lb2gt_0),81). orientation(lb2gt_3,146,81). coords(lb2gt_3,13,48,81).  visible(lb2gt_3,81). happensAt(walking(lb2gt_3),81).
orientation(lb2gt_0,141,82). coords(lb2gt_0,193,124,82).  visible(lb2gt_0,82). happensAt(active(lb2gt_0),82). orientation(lb2gt_3,146,82). coords(lb2gt_3,13,49,82).  visible(lb2gt_3,82). happensAt(walking(lb2gt_3),82).
orientation(lb2gt_0,141,83). coords(lb2gt_0,193,124,83).  visible(lb2gt_0,83). happensAt(active(lb2gt_0),83). orientation(lb2gt_3,146,83). coords(lb2gt_3,13,49,83).  visible(lb2gt_3,83). happensAt(walking(lb2gt_3),83).
orientation(lb2gt_0,141,84). coords(lb2gt_0,193,124,84).  visible(lb2gt_0,84). happensAt(active(lb2gt_0),84). orientation(lb2gt_3,146,84). coords(lb2gt_3,13,49,84).  visible(lb2gt_3,84). happensAt(walking(lb2gt_3),84).
orientation(lb2gt_0,141,85). coords(lb2gt_0,193,124,85).  visible(lb2gt_0,85). happensAt(active(lb2gt_0),85). orientation(lb2gt_3,146,85). coords(lb2gt_3,12,49,85).  visible(lb2gt_3,85). happensAt(walking(lb2gt_3),85).
orientation(lb2gt_0,141,86). coords(lb2gt_0,193,124,86).  visible(lb2gt_0,86). happensAt(active(lb2gt_0),86). orientation(lb2gt_3,146,86). coords(lb2gt_3,13,49,86).  visible(lb2gt_3,86). happensAt(walking(lb2gt_3),86).
orientation(lb2gt_0,141,87). coords(lb2gt_0,193,124,87).  visible(lb2gt_0,87). happensAt(active(lb2gt_0),87). orientation(lb2gt_3,146,87). coords(lb2gt_3,13,49,87).  visible(lb2gt_3,87). happensAt(walking(lb2gt_3),87).
orientation(lb2gt_0,141,88). coords(lb2gt_0,193,124,88).  visible(lb2gt_0,88). happensAt(active(lb2gt_0),88). orientation(lb2gt_3,146,88). coords(lb2gt_3,13,49,88).  visible(lb2gt_3,88). happensAt(walking(lb2gt_3),88).
orientation(lb2gt_0,141,89). coords(lb2gt_0,193,124,89).  visible(lb2gt_0,89). happensAt(active(lb2gt_0),89). orientation(lb2gt_3,146,89). coords(lb2gt_3,13,50,89).  visible(lb2gt_3,89). happensAt(walking(lb2gt_3),89).
orientation(lb2gt_0,141,90). coords(lb2gt_0,193,124,90).  visible(lb2gt_0,90). happensAt(active(lb2gt_0),90). orientation(lb2gt_3,146,90). coords(lb2gt_3,12,50,90).  visible(lb2gt_3,90). happensAt(walking(lb2gt_3),90).
orientation(lb2gt_0,141,91). coords(lb2gt_0,193,124,91).  visible(lb2gt_0,91). happensAt(active(lb2gt_0),91). orientation(lb2gt_3,146,91). coords(lb2gt_3,12,50,91).  visible(lb2gt_3,91). happensAt(walking(lb2gt_3),91).
orientation(lb2gt_0,141,92). coords(lb2gt_0,193,124,92).  visible(lb2gt_0,92). happensAt(active(lb2gt_0),92). orientation(lb2gt_3,146,92). coords(lb2gt_3,12,50,92).  visible(lb2gt_3,92). happensAt(walking(lb2gt_3),92).
orientation(lb2gt_0,141,93). coords(lb2gt_0,193,123,93).  visible(lb2gt_0,93). happensAt(active(lb2gt_0),93). orientation(lb2gt_3,146,93). coords(lb2gt_3,12,50,93).  visible(lb2gt_3,93). happensAt(walking(lb2gt_3),93).
orientation(lb2gt_0,141,94). coords(lb2gt_0,193,123,94).  visible(lb2gt_0,94). happensAt(active(lb2gt_0),94). orientation(lb2gt_3,146,94). coords(lb2gt_3,12,50,94).  visible(lb2gt_3,94). happensAt(walking(lb2gt_3),94).
orientation(lb2gt_0,141,95). coords(lb2gt_0,193,123,95).  visible(lb2gt_0,95). happensAt(active(lb2gt_0),95). orientation(lb2gt_3,146,95). coords(lb2gt_3,12,50,95).  visible(lb2gt_3,95). happensAt(walking(lb2gt_3),95).
orientation(lb2gt_0,141,96). coords(lb2gt_0,193,123,96).  visible(lb2gt_0,96). happensAt(active(lb2gt_0),96). orientation(lb2gt_3,146,96). coords(lb2gt_3,11,51,96).  visible(lb2gt_3,96). happensAt(walking(lb2gt_3),96).
#show holdsAt/2.