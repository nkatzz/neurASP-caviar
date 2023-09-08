
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

	orientation(ricgt_4,172,59). coords(ricgt_4,28,130,59).  visible(ricgt_4,59). happensAt(walking(ricgt_4),59). orientation(ricgt_5,0,59). coords(ricgt_5,28,122,59).  visible(ricgt_5,59). happensAt(walking(ricgt_5),59).
orientation(ricgt_4,172,60). coords(ricgt_4,28,130,60).  visible(ricgt_4,60). happensAt(walking(ricgt_4),60). orientation(ricgt_5,0,60). coords(ricgt_5,28,123,60).  visible(ricgt_5,60). happensAt(walking(ricgt_5),60).
orientation(ricgt_4,172,61). coords(ricgt_4,28,130,61).  visible(ricgt_4,61). happensAt(walking(ricgt_4),61). orientation(ricgt_5,0,61). coords(ricgt_5,28,123,61).  visible(ricgt_5,61). happensAt(walking(ricgt_5),61).
orientation(ricgt_4,172,62). coords(ricgt_4,28,130,62).  visible(ricgt_4,62). happensAt(walking(ricgt_4),62). orientation(ricgt_5,0,62). coords(ricgt_5,28,123,62).  visible(ricgt_5,62). happensAt(walking(ricgt_5),62).
orientation(ricgt_4,172,63). coords(ricgt_4,28,130,63).  visible(ricgt_4,63). happensAt(walking(ricgt_4),63). orientation(ricgt_5,0,63). coords(ricgt_5,28,123,63).  visible(ricgt_5,63). happensAt(walking(ricgt_5),63).
orientation(ricgt_4,172,64). coords(ricgt_4,28,130,64).  visible(ricgt_4,64). happensAt(walking(ricgt_4),64). orientation(ricgt_5,0,64). coords(ricgt_5,28,123,64).  visible(ricgt_5,64). happensAt(walking(ricgt_5),64).
orientation(ricgt_4,172,65). coords(ricgt_4,28,130,65).  visible(ricgt_4,65). happensAt(walking(ricgt_4),65). orientation(ricgt_5,0,65). coords(ricgt_5,28,123,65).  visible(ricgt_5,65). happensAt(walking(ricgt_5),65).
orientation(ricgt_4,172,66). coords(ricgt_4,27,131,66).  visible(ricgt_4,66). happensAt(walking(ricgt_4),66). orientation(ricgt_5,0,66). coords(ricgt_5,28,123,66).  visible(ricgt_5,66). happensAt(walking(ricgt_5),66).
orientation(ricgt_4,172,67). coords(ricgt_4,27,131,67).  visible(ricgt_4,67). happensAt(walking(ricgt_4),67). orientation(ricgt_5,0,67). coords(ricgt_5,28,123,67).  visible(ricgt_5,67). happensAt(walking(ricgt_5),67).
orientation(ricgt_4,172,68). coords(ricgt_4,27,131,68).  visible(ricgt_4,68). happensAt(walking(ricgt_4),68). orientation(ricgt_5,0,68). coords(ricgt_5,28,123,68).  visible(ricgt_5,68). happensAt(walking(ricgt_5),68).
orientation(ricgt_4,172,69). coords(ricgt_4,27,131,69).  visible(ricgt_4,69). happensAt(walking(ricgt_4),69). orientation(ricgt_5,0,69). coords(ricgt_5,28,123,69).  visible(ricgt_5,69). happensAt(walking(ricgt_5),69).
orientation(ricgt_4,172,70). coords(ricgt_4,27,131,70).  visible(ricgt_4,70). happensAt(walking(ricgt_4),70). orientation(ricgt_5,0,70). coords(ricgt_5,28,123,70).  visible(ricgt_5,70). happensAt(walking(ricgt_5),70).
orientation(ricgt_4,172,71). coords(ricgt_4,27,131,71).  visible(ricgt_4,71). happensAt(walking(ricgt_4),71). orientation(ricgt_5,0,71). coords(ricgt_5,28,123,71).  visible(ricgt_5,71). happensAt(walking(ricgt_5),71).
orientation(ricgt_4,172,72). coords(ricgt_4,27,132,72).  visible(ricgt_4,72). happensAt(walking(ricgt_4),72). orientation(ricgt_5,0,72). coords(ricgt_5,28,123,72).  visible(ricgt_5,72). happensAt(walking(ricgt_5),72).
orientation(ricgt_4,172,73). coords(ricgt_4,27,132,73).  visible(ricgt_4,73). happensAt(walking(ricgt_4),73). orientation(ricgt_5,0,73). coords(ricgt_5,28,124,73).  visible(ricgt_5,73). happensAt(walking(ricgt_5),73).
orientation(ricgt_4,172,74). coords(ricgt_4,27,132,74).  visible(ricgt_4,74). happensAt(walking(ricgt_4),74). orientation(ricgt_5,0,74). coords(ricgt_5,28,124,74).  visible(ricgt_5,74). happensAt(walking(ricgt_5),74).
orientation(ricgt_4,172,75). coords(ricgt_4,27,132,75).  visible(ricgt_4,75). happensAt(walking(ricgt_4),75). orientation(ricgt_5,0,75). coords(ricgt_5,28,124,75).  visible(ricgt_5,75). happensAt(walking(ricgt_5),75).
orientation(ricgt_4,172,76). coords(ricgt_4,27,132,76).  visible(ricgt_4,76). happensAt(walking(ricgt_4),76). orientation(ricgt_5,0,76). coords(ricgt_5,28,124,76).  visible(ricgt_5,76). happensAt(walking(ricgt_5),76).
orientation(ricgt_4,172,77). coords(ricgt_4,27,132,77).  visible(ricgt_4,77). happensAt(walking(ricgt_4),77). orientation(ricgt_5,0,77). coords(ricgt_5,28,124,77).  visible(ricgt_5,77). happensAt(walking(ricgt_5),77).
orientation(ricgt_4,172,78). coords(ricgt_4,27,132,78).  visible(ricgt_4,78). happensAt(walking(ricgt_4),78). orientation(ricgt_5,0,78). coords(ricgt_5,28,124,78).  visible(ricgt_5,78). happensAt(walking(ricgt_5),78).
orientation(ricgt_4,172,79). coords(ricgt_4,27,132,79).  visible(ricgt_4,79). happensAt(walking(ricgt_4),79). orientation(ricgt_5,0,79). coords(ricgt_5,28,124,79).  visible(ricgt_5,79). happensAt(walking(ricgt_5),79).
orientation(ricgt_4,172,80). coords(ricgt_4,27,132,80).  visible(ricgt_4,80). happensAt(walking(ricgt_4),80). orientation(ricgt_5,0,80). coords(ricgt_5,28,124,80).  visible(ricgt_5,80). happensAt(walking(ricgt_5),80).
orientation(ricgt_4,172,81). coords(ricgt_4,27,133,81).  visible(ricgt_4,81). happensAt(walking(ricgt_4),81). orientation(ricgt_5,0,81). coords(ricgt_5,28,124,81).  visible(ricgt_5,81). happensAt(walking(ricgt_5),81).
orientation(ricgt_4,172,82). coords(ricgt_4,27,133,82).  visible(ricgt_4,82). happensAt(walking(ricgt_4),82). orientation(ricgt_5,0,82). coords(ricgt_5,28,124,82).  visible(ricgt_5,82). happensAt(walking(ricgt_5),82).
orientation(ricgt_4,172,83). coords(ricgt_4,27,133,83).  visible(ricgt_4,83). happensAt(walking(ricgt_4),83). orientation(ricgt_5,0,83). coords(ricgt_5,27,125,83).  visible(ricgt_5,83). happensAt(walking(ricgt_5),83).
#show holdsAt/2.