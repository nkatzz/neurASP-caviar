
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

	orientation(lb2gt_0,141,60). coords(lb2gt_0,193,124,60).  visible(lb2gt_0,60). happensAt(active(lb2gt_0),60). orientation(lb2gt_3,146,60). coords(lb2gt_3,18,45,60).  visible(lb2gt_3,60). happensAt(walking(lb2gt_3),60).
orientation(lb2gt_0,141,61). coords(lb2gt_0,193,124,61).  visible(lb2gt_0,61). happensAt(active(lb2gt_0),61). orientation(lb2gt_3,146,61). coords(lb2gt_3,18,45,61).  visible(lb2gt_3,61). happensAt(walking(lb2gt_3),61).
orientation(lb2gt_0,141,62). coords(lb2gt_0,193,124,62).  visible(lb2gt_0,62). happensAt(active(lb2gt_0),62). orientation(lb2gt_3,146,62). coords(lb2gt_3,17,46,62).  visible(lb2gt_3,62). happensAt(walking(lb2gt_3),62).
orientation(lb2gt_0,141,63). coords(lb2gt_0,193,124,63).  visible(lb2gt_0,63). happensAt(active(lb2gt_0),63). orientation(lb2gt_3,146,63). coords(lb2gt_3,17,46,63).  visible(lb2gt_3,63). happensAt(walking(lb2gt_3),63).
orientation(lb2gt_0,141,64). coords(lb2gt_0,193,124,64).  visible(lb2gt_0,64). happensAt(active(lb2gt_0),64). orientation(lb2gt_3,146,64). coords(lb2gt_3,17,46,64).  visible(lb2gt_3,64). happensAt(walking(lb2gt_3),64).
orientation(lb2gt_0,141,65). coords(lb2gt_0,193,124,65).  visible(lb2gt_0,65). happensAt(active(lb2gt_0),65). orientation(lb2gt_3,146,65). coords(lb2gt_3,17,46,65).  visible(lb2gt_3,65). happensAt(walking(lb2gt_3),65).
orientation(lb2gt_0,141,66). coords(lb2gt_0,193,124,66).  visible(lb2gt_0,66). happensAt(active(lb2gt_0),66). orientation(lb2gt_3,146,66). coords(lb2gt_3,16,46,66).  visible(lb2gt_3,66). happensAt(walking(lb2gt_3),66).
orientation(lb2gt_0,141,67). coords(lb2gt_0,193,124,67).  visible(lb2gt_0,67). happensAt(active(lb2gt_0),67). orientation(lb2gt_3,146,67). coords(lb2gt_3,16,46,67).  visible(lb2gt_3,67). happensAt(walking(lb2gt_3),67).
orientation(lb2gt_0,141,68). coords(lb2gt_0,193,124,68).  visible(lb2gt_0,68). happensAt(active(lb2gt_0),68). orientation(lb2gt_3,146,68). coords(lb2gt_3,15,46,68).  visible(lb2gt_3,68). happensAt(walking(lb2gt_3),68).
orientation(lb2gt_0,141,69). coords(lb2gt_0,193,124,69).  visible(lb2gt_0,69). happensAt(active(lb2gt_0),69). orientation(lb2gt_3,146,69). coords(lb2gt_3,15,46,69).  visible(lb2gt_3,69). happensAt(walking(lb2gt_3),69).
orientation(lb2gt_0,141,70). coords(lb2gt_0,193,124,70).  visible(lb2gt_0,70). happensAt(active(lb2gt_0),70). orientation(lb2gt_3,146,70). coords(lb2gt_3,15,46,70).  visible(lb2gt_3,70). happensAt(walking(lb2gt_3),70).
orientation(lb2gt_0,141,71). coords(lb2gt_0,193,124,71).  visible(lb2gt_0,71). happensAt(active(lb2gt_0),71). orientation(lb2gt_3,146,71). coords(lb2gt_3,15,46,71).  visible(lb2gt_3,71). happensAt(walking(lb2gt_3),71).
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
#show holdsAt/2.