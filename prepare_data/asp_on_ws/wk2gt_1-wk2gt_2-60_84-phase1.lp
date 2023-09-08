
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

	orientation(wk2gt_1,140,60). coords(wk2gt_1,112,55,60).  visible(wk2gt_1,60). happensAt(walking(wk2gt_1),60). orientation(wk2gt_2,141,60). coords(wk2gt_2,98,63,60).  visible(wk2gt_2,60). happensAt(walking(wk2gt_2),60).
orientation(wk2gt_1,140,61). coords(wk2gt_1,112,54,61).  visible(wk2gt_1,61). happensAt(walking(wk2gt_1),61). orientation(wk2gt_2,141,61). coords(wk2gt_2,98,63,61).  visible(wk2gt_2,61). happensAt(walking(wk2gt_2),61).
orientation(wk2gt_1,139,62). coords(wk2gt_1,112,53,62).  visible(wk2gt_1,62). happensAt(walking(wk2gt_1),62). orientation(wk2gt_2,141,62). coords(wk2gt_2,98,63,62).  visible(wk2gt_2,62). happensAt(walking(wk2gt_2),62).
orientation(wk2gt_1,140,63). coords(wk2gt_1,112,53,63).  visible(wk2gt_1,63). happensAt(walking(wk2gt_1),63). orientation(wk2gt_2,141,63). coords(wk2gt_2,98,63,63).  visible(wk2gt_2,63). happensAt(walking(wk2gt_2),63).
orientation(wk2gt_1,134,64). coords(wk2gt_1,112,53,64).  visible(wk2gt_1,64). happensAt(active(wk2gt_1),64). orientation(wk2gt_2,141,64). coords(wk2gt_2,98,63,64).  visible(wk2gt_2,64). happensAt(active(wk2gt_2),64).
orientation(wk2gt_1,135,65). coords(wk2gt_1,112,53,65).  visible(wk2gt_1,65). happensAt(active(wk2gt_1),65). orientation(wk2gt_2,141,65). coords(wk2gt_2,98,63,65).  visible(wk2gt_2,65). happensAt(active(wk2gt_2),65).
orientation(wk2gt_1,134,66). coords(wk2gt_1,113,53,66).  visible(wk2gt_1,66). happensAt(active(wk2gt_1),66). orientation(wk2gt_2,141,66). coords(wk2gt_2,98,63,66).  visible(wk2gt_2,66). happensAt(active(wk2gt_2),66).
orientation(wk2gt_1,142,67). coords(wk2gt_1,113,53,67).  visible(wk2gt_1,67). happensAt(active(wk2gt_1),67). orientation(wk2gt_2,141,67). coords(wk2gt_2,98,63,67).  visible(wk2gt_2,67). happensAt(active(wk2gt_2),67).
orientation(wk2gt_1,135,68). coords(wk2gt_1,113,53,68).  visible(wk2gt_1,68). happensAt(active(wk2gt_1),68). orientation(wk2gt_2,141,68). coords(wk2gt_2,98,63,68).  visible(wk2gt_2,68). happensAt(active(wk2gt_2),68).
orientation(wk2gt_1,135,69). coords(wk2gt_1,113,53,69).  visible(wk2gt_1,69). happensAt(active(wk2gt_1),69). orientation(wk2gt_2,141,69). coords(wk2gt_2,98,63,69).  visible(wk2gt_2,69). happensAt(active(wk2gt_2),69).
orientation(wk2gt_1,135,70). coords(wk2gt_1,113,53,70).  visible(wk2gt_1,70). happensAt(active(wk2gt_1),70). orientation(wk2gt_2,141,70). coords(wk2gt_2,98,63,70).  visible(wk2gt_2,70). happensAt(active(wk2gt_2),70).
orientation(wk2gt_1,135,71). coords(wk2gt_1,114,53,71).  visible(wk2gt_1,71). happensAt(active(wk2gt_1),71). orientation(wk2gt_2,141,71). coords(wk2gt_2,98,63,71).  visible(wk2gt_2,71). happensAt(active(wk2gt_2),71).
orientation(wk2gt_1,135,72). coords(wk2gt_1,114,53,72).  visible(wk2gt_1,72). happensAt(active(wk2gt_1),72). orientation(wk2gt_2,141,72). coords(wk2gt_2,98,63,72).  visible(wk2gt_2,72). happensAt(active(wk2gt_2),72).
orientation(wk2gt_1,135,73). coords(wk2gt_1,114,53,73).  visible(wk2gt_1,73). happensAt(active(wk2gt_1),73). orientation(wk2gt_2,141,73). coords(wk2gt_2,98,63,73).  visible(wk2gt_2,73). happensAt(active(wk2gt_2),73).
orientation(wk2gt_1,138,74). coords(wk2gt_1,114,53,74).  visible(wk2gt_1,74). happensAt(active(wk2gt_1),74). orientation(wk2gt_2,141,74). coords(wk2gt_2,98,63,74).  visible(wk2gt_2,74). happensAt(active(wk2gt_2),74).
orientation(wk2gt_1,137,75). coords(wk2gt_1,114,53,75).  visible(wk2gt_1,75). happensAt(inactive(wk2gt_1),75). orientation(wk2gt_2,141,75). coords(wk2gt_2,98,63,75).  visible(wk2gt_2,75). happensAt(active(wk2gt_2),75).
orientation(wk2gt_1,137,76). coords(wk2gt_1,115,53,76).  visible(wk2gt_1,76). happensAt(inactive(wk2gt_1),76). orientation(wk2gt_2,141,76). coords(wk2gt_2,98,63,76).  visible(wk2gt_2,76). happensAt(active(wk2gt_2),76).
orientation(wk2gt_1,136,77). coords(wk2gt_1,115,53,77).  visible(wk2gt_1,77). happensAt(inactive(wk2gt_1),77). orientation(wk2gt_2,141,77). coords(wk2gt_2,98,63,77).  visible(wk2gt_2,77). happensAt(active(wk2gt_2),77).
orientation(wk2gt_1,141,78). coords(wk2gt_1,114,53,78).  visible(wk2gt_1,78). happensAt(inactive(wk2gt_1),78). orientation(wk2gt_2,141,78). coords(wk2gt_2,98,63,78).  visible(wk2gt_2,78). happensAt(active(wk2gt_2),78).
orientation(wk2gt_1,139,79). coords(wk2gt_1,114,53,79).  visible(wk2gt_1,79). happensAt(inactive(wk2gt_1),79). orientation(wk2gt_2,141,79). coords(wk2gt_2,98,63,79).  visible(wk2gt_2,79). happensAt(active(wk2gt_2),79).
orientation(wk2gt_1,135,80). coords(wk2gt_1,113,53,80).  visible(wk2gt_1,80). happensAt(inactive(wk2gt_1),80). orientation(wk2gt_2,141,80). coords(wk2gt_2,98,63,80).  visible(wk2gt_2,80). happensAt(active(wk2gt_2),80).
orientation(wk2gt_1,134,81). coords(wk2gt_1,113,53,81).  visible(wk2gt_1,81). happensAt(inactive(wk2gt_1),81). orientation(wk2gt_2,141,81). coords(wk2gt_2,98,63,81).  visible(wk2gt_2,81). happensAt(active(wk2gt_2),81).
orientation(wk2gt_1,134,82). coords(wk2gt_1,112,53,82).  visible(wk2gt_1,82). happensAt(inactive(wk2gt_1),82). orientation(wk2gt_2,141,82). coords(wk2gt_2,98,63,82).  visible(wk2gt_2,82). happensAt(active(wk2gt_2),82).
orientation(wk2gt_1,139,83). coords(wk2gt_1,112,53,83).  visible(wk2gt_1,83). happensAt(inactive(wk2gt_1),83). orientation(wk2gt_2,141,83). coords(wk2gt_2,98,63,83).  visible(wk2gt_2,83). happensAt(active(wk2gt_2),83).
orientation(wk2gt_1,137,84). coords(wk2gt_1,113,53,84).  visible(wk2gt_1,84). happensAt(inactive(wk2gt_1),84). orientation(wk2gt_2,141,84). coords(wk2gt_2,98,63,84).  visible(wk2gt_2,84). happensAt(active(wk2gt_2),84).
#show holdsAt/2.