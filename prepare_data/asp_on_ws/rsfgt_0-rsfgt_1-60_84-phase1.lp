
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

	orientation(rsfgt_0,160,60). coords(rsfgt_0,79,136,60).  visible(rsfgt_0,60). happensAt(walking(rsfgt_0),60). orientation(rsfgt_1,150,60). coords(rsfgt_1,177,117,60).  visible(rsfgt_1,60). happensAt(walking(rsfgt_1),60).
orientation(rsfgt_0,160,61). coords(rsfgt_0,78,137,61).  visible(rsfgt_0,61). happensAt(walking(rsfgt_0),61). orientation(rsfgt_1,150,61). coords(rsfgt_1,176,117,61).  visible(rsfgt_1,61). happensAt(walking(rsfgt_1),61).
orientation(rsfgt_0,160,62). coords(rsfgt_0,78,137,62).  visible(rsfgt_0,62). happensAt(walking(rsfgt_0),62). orientation(rsfgt_1,150,62). coords(rsfgt_1,175,117,62).  visible(rsfgt_1,62). happensAt(walking(rsfgt_1),62).
orientation(rsfgt_0,160,63). coords(rsfgt_0,78,137,63).  visible(rsfgt_0,63). happensAt(walking(rsfgt_0),63). orientation(rsfgt_1,150,63). coords(rsfgt_1,174,117,63).  visible(rsfgt_1,63). happensAt(walking(rsfgt_1),63).
orientation(rsfgt_0,160,64). coords(rsfgt_0,75,136,64).  visible(rsfgt_0,64). happensAt(walking(rsfgt_0),64). orientation(rsfgt_1,150,64). coords(rsfgt_1,174,117,64).  visible(rsfgt_1,64). happensAt(walking(rsfgt_1),64).
orientation(rsfgt_0,160,65). coords(rsfgt_0,73,137,65).  visible(rsfgt_0,65). happensAt(walking(rsfgt_0),65). orientation(rsfgt_1,150,65). coords(rsfgt_1,174,117,65).  visible(rsfgt_1,65). happensAt(walking(rsfgt_1),65).
orientation(rsfgt_0,160,66). coords(rsfgt_0,72,138,66).  visible(rsfgt_0,66). happensAt(walking(rsfgt_0),66). orientation(rsfgt_1,150,66). coords(rsfgt_1,174,117,66).  visible(rsfgt_1,66). happensAt(walking(rsfgt_1),66).
orientation(rsfgt_0,160,67). coords(rsfgt_0,71,139,67).  visible(rsfgt_0,67). happensAt(walking(rsfgt_0),67). orientation(rsfgt_1,150,67). coords(rsfgt_1,174,117,67).  visible(rsfgt_1,67). happensAt(walking(rsfgt_1),67).
orientation(rsfgt_0,160,68). coords(rsfgt_0,71,139,68).  visible(rsfgt_0,68). happensAt(walking(rsfgt_0),68). orientation(rsfgt_1,150,68). coords(rsfgt_1,173,117,68).  visible(rsfgt_1,68). happensAt(walking(rsfgt_1),68).
orientation(rsfgt_0,160,69). coords(rsfgt_0,71,139,69).  visible(rsfgt_0,69). happensAt(walking(rsfgt_0),69). orientation(rsfgt_1,150,69). coords(rsfgt_1,173,118,69).  visible(rsfgt_1,69). happensAt(walking(rsfgt_1),69).
orientation(rsfgt_0,160,70). coords(rsfgt_0,71,139,70).  visible(rsfgt_0,70). happensAt(walking(rsfgt_0),70). orientation(rsfgt_1,150,70). coords(rsfgt_1,172,117,70).  visible(rsfgt_1,70). happensAt(walking(rsfgt_1),70).
orientation(rsfgt_0,160,71). coords(rsfgt_0,71,140,71).  visible(rsfgt_0,71). happensAt(walking(rsfgt_0),71). orientation(rsfgt_1,150,71). coords(rsfgt_1,170,116,71).  visible(rsfgt_1,71). happensAt(walking(rsfgt_1),71).
orientation(rsfgt_0,160,72). coords(rsfgt_0,70,141,72).  visible(rsfgt_0,72). happensAt(walking(rsfgt_0),72). orientation(rsfgt_1,150,72). coords(rsfgt_1,170,116,72).  visible(rsfgt_1,72). happensAt(walking(rsfgt_1),72).
orientation(rsfgt_0,160,73). coords(rsfgt_0,70,142,73).  visible(rsfgt_0,73). happensAt(walking(rsfgt_0),73). orientation(rsfgt_1,150,73). coords(rsfgt_1,170,116,73).  visible(rsfgt_1,73). happensAt(walking(rsfgt_1),73).
orientation(rsfgt_0,160,74). coords(rsfgt_0,69,143,74).  visible(rsfgt_0,74). happensAt(walking(rsfgt_0),74). orientation(rsfgt_1,150,74). coords(rsfgt_1,170,116,74).  visible(rsfgt_1,74). happensAt(walking(rsfgt_1),74).
orientation(rsfgt_0,160,75). coords(rsfgt_0,68,143,75).  visible(rsfgt_0,75). happensAt(walking(rsfgt_0),75). orientation(rsfgt_1,150,75). coords(rsfgt_1,170,115,75).  visible(rsfgt_1,75). happensAt(walking(rsfgt_1),75).
orientation(rsfgt_0,160,76). coords(rsfgt_0,67,143,76).  visible(rsfgt_0,76). happensAt(walking(rsfgt_0),76). orientation(rsfgt_1,150,76). coords(rsfgt_1,170,115,76).  visible(rsfgt_1,76). happensAt(walking(rsfgt_1),76).
orientation(rsfgt_0,160,77). coords(rsfgt_0,66,143,77).  visible(rsfgt_0,77). happensAt(walking(rsfgt_0),77). orientation(rsfgt_1,150,77). coords(rsfgt_1,168,113,77).  visible(rsfgt_1,77). happensAt(walking(rsfgt_1),77).
orientation(rsfgt_0,160,78). coords(rsfgt_0,65,143,78).  visible(rsfgt_0,78). happensAt(walking(rsfgt_0),78). orientation(rsfgt_1,150,78). coords(rsfgt_1,168,113,78).  visible(rsfgt_1,78). happensAt(walking(rsfgt_1),78).
orientation(rsfgt_0,160,79). coords(rsfgt_0,64,143,79).  visible(rsfgt_0,79). happensAt(walking(rsfgt_0),79). orientation(rsfgt_1,150,79). coords(rsfgt_1,168,113,79).  visible(rsfgt_1,79). happensAt(walking(rsfgt_1),79).
orientation(rsfgt_0,160,80). coords(rsfgt_0,62,143,80).  visible(rsfgt_0,80). happensAt(walking(rsfgt_0),80). orientation(rsfgt_1,150,80). coords(rsfgt_1,168,113,80).  visible(rsfgt_1,80). happensAt(walking(rsfgt_1),80).
orientation(rsfgt_0,160,81). coords(rsfgt_0,62,143,81).  visible(rsfgt_0,81). happensAt(walking(rsfgt_0),81). orientation(rsfgt_1,150,81). coords(rsfgt_1,168,113,81).  visible(rsfgt_1,81). happensAt(walking(rsfgt_1),81).
orientation(rsfgt_0,160,82). coords(rsfgt_0,62,144,82).  visible(rsfgt_0,82). happensAt(walking(rsfgt_0),82). orientation(rsfgt_1,150,82). coords(rsfgt_1,168,113,82).  visible(rsfgt_1,82). happensAt(walking(rsfgt_1),82).
orientation(rsfgt_0,160,83). coords(rsfgt_0,62,144,83).  visible(rsfgt_0,83). happensAt(walking(rsfgt_0),83). orientation(rsfgt_1,150,83). coords(rsfgt_1,168,112,83).  visible(rsfgt_1,83). happensAt(walking(rsfgt_1),83).
orientation(rsfgt_0,160,84). coords(rsfgt_0,61,144,84).  visible(rsfgt_0,84). happensAt(walking(rsfgt_0),84). orientation(rsfgt_1,150,84). coords(rsfgt_1,166,112,84).  visible(rsfgt_1,84). happensAt(walking(rsfgt_1),84).
#show holdsAt/2.