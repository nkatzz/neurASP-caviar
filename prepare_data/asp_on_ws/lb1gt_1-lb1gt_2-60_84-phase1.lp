
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

	orientation(lb1gt_1,147,60). coords(lb1gt_1,74,101,60).  visible(lb1gt_1,60). happensAt(walking(lb1gt_1),60). orientation(lb1gt_2,142,60). coords(lb1gt_2,83,79,60).  visible(lb1gt_2,60). happensAt(walking(lb1gt_2),60).
orientation(lb1gt_1,147,61). coords(lb1gt_1,74,101,61).  visible(lb1gt_1,61). happensAt(walking(lb1gt_1),61). orientation(lb1gt_2,142,61). coords(lb1gt_2,83,80,61).  visible(lb1gt_2,61). happensAt(walking(lb1gt_2),61).
orientation(lb1gt_1,147,62). coords(lb1gt_1,74,102,62).  visible(lb1gt_1,62). happensAt(walking(lb1gt_1),62). orientation(lb1gt_2,142,62). coords(lb1gt_2,83,80,62).  visible(lb1gt_2,62). happensAt(walking(lb1gt_2),62).
orientation(lb1gt_1,147,63). coords(lb1gt_1,74,102,63).  visible(lb1gt_1,63). happensAt(walking(lb1gt_1),63). orientation(lb1gt_2,142,63). coords(lb1gt_2,83,80,63).  visible(lb1gt_2,63). happensAt(walking(lb1gt_2),63).
orientation(lb1gt_1,147,64). coords(lb1gt_1,74,102,64).  visible(lb1gt_1,64). happensAt(walking(lb1gt_1),64). orientation(lb1gt_2,142,64). coords(lb1gt_2,83,80,64).  visible(lb1gt_2,64). happensAt(walking(lb1gt_2),64).
orientation(lb1gt_1,147,65). coords(lb1gt_1,74,102,65).  visible(lb1gt_1,65). happensAt(walking(lb1gt_1),65). orientation(lb1gt_2,142,65). coords(lb1gt_2,84,80,65).  visible(lb1gt_2,65). happensAt(walking(lb1gt_2),65).
orientation(lb1gt_1,147,66). coords(lb1gt_1,74,102,66).  visible(lb1gt_1,66). happensAt(walking(lb1gt_1),66). orientation(lb1gt_2,142,66). coords(lb1gt_2,84,81,66).  visible(lb1gt_2,66). happensAt(walking(lb1gt_2),66).
orientation(lb1gt_1,147,67). coords(lb1gt_1,74,103,67).  visible(lb1gt_1,67). happensAt(walking(lb1gt_1),67). orientation(lb1gt_2,142,67). coords(lb1gt_2,84,81,67).  visible(lb1gt_2,67). happensAt(walking(lb1gt_2),67).
orientation(lb1gt_1,147,68). coords(lb1gt_1,74,103,68).  visible(lb1gt_1,68). happensAt(walking(lb1gt_1),68). orientation(lb1gt_2,142,68). coords(lb1gt_2,85,81,68).  visible(lb1gt_2,68). happensAt(walking(lb1gt_2),68).
orientation(lb1gt_1,147,69). coords(lb1gt_1,73,103,69).  visible(lb1gt_1,69). happensAt(walking(lb1gt_1),69). orientation(lb1gt_2,142,69). coords(lb1gt_2,85,81,69).  visible(lb1gt_2,69). happensAt(walking(lb1gt_2),69).
orientation(lb1gt_1,147,70). coords(lb1gt_1,73,103,70).  visible(lb1gt_1,70). happensAt(walking(lb1gt_1),70). orientation(lb1gt_2,142,70). coords(lb1gt_2,86,81,70).  visible(lb1gt_2,70). happensAt(walking(lb1gt_2),70).
orientation(lb1gt_1,147,71). coords(lb1gt_1,74,104,71).  visible(lb1gt_1,71). happensAt(walking(lb1gt_1),71). orientation(lb1gt_2,142,71). coords(lb1gt_2,86,81,71).  visible(lb1gt_2,71). happensAt(walking(lb1gt_2),71).
orientation(lb1gt_1,147,72). coords(lb1gt_1,74,104,72).  visible(lb1gt_1,72). happensAt(walking(lb1gt_1),72). orientation(lb1gt_2,142,72). coords(lb1gt_2,87,81,72).  visible(lb1gt_2,72). happensAt(walking(lb1gt_2),72).
orientation(lb1gt_1,147,73). coords(lb1gt_1,75,104,73).  visible(lb1gt_1,73). happensAt(walking(lb1gt_1),73). orientation(lb1gt_2,142,73). coords(lb1gt_2,87,81,73).  visible(lb1gt_2,73). happensAt(walking(lb1gt_2),73).
orientation(lb1gt_1,147,74). coords(lb1gt_1,75,105,74).  visible(lb1gt_1,74). happensAt(walking(lb1gt_1),74). orientation(lb1gt_2,142,74). coords(lb1gt_2,87,81,74).  visible(lb1gt_2,74). happensAt(walking(lb1gt_2),74).
orientation(lb1gt_1,147,75). coords(lb1gt_1,75,105,75).  visible(lb1gt_1,75). happensAt(walking(lb1gt_1),75). orientation(lb1gt_2,142,75). coords(lb1gt_2,88,81,75).  visible(lb1gt_2,75). happensAt(walking(lb1gt_2),75).
orientation(lb1gt_1,147,76). coords(lb1gt_1,75,106,76).  visible(lb1gt_1,76). happensAt(walking(lb1gt_1),76). orientation(lb1gt_2,142,76). coords(lb1gt_2,88,81,76).  visible(lb1gt_2,76). happensAt(walking(lb1gt_2),76).
orientation(lb1gt_1,147,77). coords(lb1gt_1,75,107,77).  visible(lb1gt_1,77). happensAt(walking(lb1gt_1),77). orientation(lb1gt_2,142,77). coords(lb1gt_2,88,81,77).  visible(lb1gt_2,77). happensAt(walking(lb1gt_2),77).
orientation(lb1gt_1,147,78). coords(lb1gt_1,75,107,78).  visible(lb1gt_1,78). happensAt(walking(lb1gt_1),78). orientation(lb1gt_2,142,78). coords(lb1gt_2,88,81,78).  visible(lb1gt_2,78). happensAt(walking(lb1gt_2),78).
orientation(lb1gt_1,147,79). coords(lb1gt_1,75,108,79).  visible(lb1gt_1,79). happensAt(walking(lb1gt_1),79). orientation(lb1gt_2,142,79). coords(lb1gt_2,88,81,79).  visible(lb1gt_2,79). happensAt(walking(lb1gt_2),79).
orientation(lb1gt_1,147,80). coords(lb1gt_1,75,109,80).  visible(lb1gt_1,80). happensAt(walking(lb1gt_1),80). orientation(lb1gt_2,142,80). coords(lb1gt_2,88,82,80).  visible(lb1gt_2,80). happensAt(walking(lb1gt_2),80).
orientation(lb1gt_1,147,81). coords(lb1gt_1,75,109,81).  visible(lb1gt_1,81). happensAt(walking(lb1gt_1),81). orientation(lb1gt_2,142,81). coords(lb1gt_2,88,82,81).  visible(lb1gt_2,81). happensAt(walking(lb1gt_2),81).
orientation(lb1gt_1,147,82). coords(lb1gt_1,75,109,82).  visible(lb1gt_1,82). happensAt(walking(lb1gt_1),82). orientation(lb1gt_2,142,82). coords(lb1gt_2,88,83,82).  visible(lb1gt_2,82). happensAt(walking(lb1gt_2),82).
orientation(lb1gt_1,147,83). coords(lb1gt_1,75,109,83).  visible(lb1gt_1,83). happensAt(walking(lb1gt_1),83). orientation(lb1gt_2,142,83). coords(lb1gt_2,88,83,83).  visible(lb1gt_2,83). happensAt(walking(lb1gt_2),83).
orientation(lb1gt_1,147,84). coords(lb1gt_1,75,109,84).  visible(lb1gt_1,84). happensAt(walking(lb1gt_1),84). orientation(lb1gt_2,142,84). coords(lb1gt_2,88,84,84).  visible(lb1gt_2,84). happensAt(walking(lb1gt_2),84).
#show holdsAt/2.