
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

	orientation(fcgt_0,143,60). coords(fcgt_0,198,150,60).  visible(fcgt_0,60). happensAt(walking(fcgt_0),60). orientation(fcgt_3,159,60). coords(fcgt_3,103,163,60).  visible(fcgt_3,60). happensAt(inactive(fcgt_3),60).
orientation(fcgt_0,143,61). coords(fcgt_0,198,150,61).  visible(fcgt_0,61). happensAt(walking(fcgt_0),61). orientation(fcgt_3,159,61). coords(fcgt_3,103,163,61).  visible(fcgt_3,61). happensAt(inactive(fcgt_3),61).
orientation(fcgt_0,143,62). coords(fcgt_0,199,151,62).  visible(fcgt_0,62). happensAt(walking(fcgt_0),62). orientation(fcgt_3,159,62). coords(fcgt_3,103,163,62).  visible(fcgt_3,62). happensAt(inactive(fcgt_3),62).
orientation(fcgt_0,143,63). coords(fcgt_0,199,151,63).  visible(fcgt_0,63). happensAt(walking(fcgt_0),63). orientation(fcgt_3,159,63). coords(fcgt_3,103,163,63).  visible(fcgt_3,63). happensAt(inactive(fcgt_3),63).
orientation(fcgt_0,143,64). coords(fcgt_0,199,151,64).  visible(fcgt_0,64). happensAt(walking(fcgt_0),64). orientation(fcgt_3,159,64). coords(fcgt_3,103,163,64).  visible(fcgt_3,64). happensAt(inactive(fcgt_3),64).
orientation(fcgt_0,143,65). coords(fcgt_0,199,151,65).  visible(fcgt_0,65). happensAt(inactive(fcgt_0),65). orientation(fcgt_3,159,65). coords(fcgt_3,103,163,65).  visible(fcgt_3,65). happensAt(inactive(fcgt_3),65).
orientation(fcgt_0,143,66). coords(fcgt_0,199,151,66).  visible(fcgt_0,66). happensAt(inactive(fcgt_0),66). orientation(fcgt_3,159,66). coords(fcgt_3,103,163,66).  visible(fcgt_3,66). happensAt(inactive(fcgt_3),66).
orientation(fcgt_0,143,67). coords(fcgt_0,199,151,67).  visible(fcgt_0,67). happensAt(inactive(fcgt_0),67). orientation(fcgt_3,159,67). coords(fcgt_3,103,163,67).  visible(fcgt_3,67). happensAt(inactive(fcgt_3),67).
orientation(fcgt_0,143,68). coords(fcgt_0,199,151,68).  visible(fcgt_0,68). happensAt(inactive(fcgt_0),68). orientation(fcgt_3,159,68). coords(fcgt_3,103,163,68).  visible(fcgt_3,68). happensAt(inactive(fcgt_3),68).
orientation(fcgt_0,143,69). coords(fcgt_0,199,151,69).  visible(fcgt_0,69). happensAt(inactive(fcgt_0),69). orientation(fcgt_3,159,69). coords(fcgt_3,103,163,69).  visible(fcgt_3,69). happensAt(inactive(fcgt_3),69).
orientation(fcgt_0,143,70). coords(fcgt_0,199,151,70).  visible(fcgt_0,70). happensAt(inactive(fcgt_0),70). orientation(fcgt_3,159,70). coords(fcgt_3,103,163,70).  visible(fcgt_3,70). happensAt(inactive(fcgt_3),70).
orientation(fcgt_0,143,71). coords(fcgt_0,199,151,71).  visible(fcgt_0,71). happensAt(inactive(fcgt_0),71). orientation(fcgt_3,159,71). coords(fcgt_3,103,163,71).  visible(fcgt_3,71). happensAt(inactive(fcgt_3),71).
orientation(fcgt_0,143,72). coords(fcgt_0,199,151,72).  visible(fcgt_0,72). happensAt(inactive(fcgt_0),72). orientation(fcgt_3,159,72). coords(fcgt_3,103,163,72).  visible(fcgt_3,72). happensAt(inactive(fcgt_3),72).
orientation(fcgt_0,143,73). coords(fcgt_0,199,151,73).  visible(fcgt_0,73). happensAt(inactive(fcgt_0),73). orientation(fcgt_3,159,73). coords(fcgt_3,103,163,73).  visible(fcgt_3,73). happensAt(inactive(fcgt_3),73).
orientation(fcgt_0,143,74). coords(fcgt_0,199,151,74).  visible(fcgt_0,74). happensAt(inactive(fcgt_0),74). orientation(fcgt_3,159,74). coords(fcgt_3,103,163,74).  visible(fcgt_3,74). happensAt(inactive(fcgt_3),74).
orientation(fcgt_0,143,75). coords(fcgt_0,200,151,75).  visible(fcgt_0,75). happensAt(inactive(fcgt_0),75). orientation(fcgt_3,159,75). coords(fcgt_3,103,163,75).  visible(fcgt_3,75). happensAt(inactive(fcgt_3),75).
orientation(fcgt_0,143,76). coords(fcgt_0,200,151,76).  visible(fcgt_0,76). happensAt(inactive(fcgt_0),76). orientation(fcgt_3,159,76). coords(fcgt_3,103,163,76).  visible(fcgt_3,76). happensAt(inactive(fcgt_3),76).
orientation(fcgt_0,143,77). coords(fcgt_0,200,151,77).  visible(fcgt_0,77). happensAt(inactive(fcgt_0),77). orientation(fcgt_3,159,77). coords(fcgt_3,103,163,77).  visible(fcgt_3,77). happensAt(inactive(fcgt_3),77).
orientation(fcgt_0,143,78). coords(fcgt_0,200,151,78).  visible(fcgt_0,78). happensAt(inactive(fcgt_0),78). orientation(fcgt_3,159,78). coords(fcgt_3,103,163,78).  visible(fcgt_3,78). happensAt(inactive(fcgt_3),78).
orientation(fcgt_0,143,79). coords(fcgt_0,200,151,79).  visible(fcgt_0,79). happensAt(inactive(fcgt_0),79). orientation(fcgt_3,159,79). coords(fcgt_3,103,163,79).  visible(fcgt_3,79). happensAt(inactive(fcgt_3),79).
orientation(fcgt_0,143,80). coords(fcgt_0,200,152,80).  visible(fcgt_0,80). happensAt(inactive(fcgt_0),80). orientation(fcgt_3,159,80). coords(fcgt_3,103,163,80).  visible(fcgt_3,80). happensAt(inactive(fcgt_3),80).
orientation(fcgt_0,143,81). coords(fcgt_0,200,152,81).  visible(fcgt_0,81). happensAt(inactive(fcgt_0),81). orientation(fcgt_3,159,81). coords(fcgt_3,103,163,81).  visible(fcgt_3,81). happensAt(inactive(fcgt_3),81).
orientation(fcgt_0,143,82). coords(fcgt_0,201,153,82).  visible(fcgt_0,82). happensAt(inactive(fcgt_0),82). orientation(fcgt_3,159,82). coords(fcgt_3,103,163,82).  visible(fcgt_3,82). happensAt(inactive(fcgt_3),82).
orientation(fcgt_0,147,83). coords(fcgt_0,201,153,83).  visible(fcgt_0,83). happensAt(walking(fcgt_0),83). orientation(fcgt_3,159,83). coords(fcgt_3,103,163,83).  visible(fcgt_3,83). happensAt(inactive(fcgt_3),83).
orientation(fcgt_0,147,84). coords(fcgt_0,202,154,84).  visible(fcgt_0,84). happensAt(walking(fcgt_0),84). orientation(fcgt_3,159,84). coords(fcgt_3,103,163,84).  visible(fcgt_3,84). happensAt(inactive(fcgt_3),84).
#show holdsAt/2.