
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

	orientation(rffgt_0,137,60). coords(rffgt_0,136,61,60).  visible(rffgt_0,60). happensAt(active(rffgt_0),60). orientation(rffgt_1,148,60). coords(rffgt_1,157,108,60).  visible(rffgt_1,60). happensAt(active(rffgt_1),60).
orientation(rffgt_0,137,61). coords(rffgt_0,136,61,61).  visible(rffgt_0,61). happensAt(active(rffgt_0),61). orientation(rffgt_1,148,61). coords(rffgt_1,155,109,61).  visible(rffgt_1,61). happensAt(active(rffgt_1),61).
orientation(rffgt_0,137,62). coords(rffgt_0,136,62,62).  visible(rffgt_0,62). happensAt(active(rffgt_0),62). orientation(rffgt_1,144,62). coords(rffgt_1,155,111,62).  visible(rffgt_1,62). happensAt(active(rffgt_1),62).
orientation(rffgt_0,137,63). coords(rffgt_0,136,62,63).  visible(rffgt_0,63). happensAt(active(rffgt_0),63). orientation(rffgt_1,144,63). coords(rffgt_1,155,112,63).  visible(rffgt_1,63). happensAt(active(rffgt_1),63).
orientation(rffgt_0,137,64). coords(rffgt_0,136,62,64).  visible(rffgt_0,64). happensAt(active(rffgt_0),64). orientation(rffgt_1,144,64). coords(rffgt_1,154,112,64).  visible(rffgt_1,64). happensAt(active(rffgt_1),64).
orientation(rffgt_0,137,65). coords(rffgt_0,136,62,65).  visible(rffgt_0,65). happensAt(active(rffgt_0),65). orientation(rffgt_1,139,65). coords(rffgt_1,155,112,65).  visible(rffgt_1,65). happensAt(active(rffgt_1),65).
orientation(rffgt_0,137,66). coords(rffgt_0,136,61,66).  visible(rffgt_0,66). happensAt(active(rffgt_0),66). orientation(rffgt_1,139,66). coords(rffgt_1,155,113,66).  visible(rffgt_1,66). happensAt(active(rffgt_1),66).
orientation(rffgt_0,137,67). coords(rffgt_0,136,61,67).  visible(rffgt_0,67). happensAt(active(rffgt_0),67). orientation(rffgt_1,139,67). coords(rffgt_1,155,113,67).  visible(rffgt_1,67). happensAt(active(rffgt_1),67).
orientation(rffgt_0,137,68). coords(rffgt_0,136,61,68).  visible(rffgt_0,68). happensAt(active(rffgt_0),68). orientation(rffgt_1,136,68). coords(rffgt_1,155,113,68).  visible(rffgt_1,68). happensAt(active(rffgt_1),68).
orientation(rffgt_0,137,69). coords(rffgt_0,135,61,69).  visible(rffgt_0,69). happensAt(active(rffgt_0),69). orientation(rffgt_1,136,69). coords(rffgt_1,155,113,69).  visible(rffgt_1,69). happensAt(active(rffgt_1),69).
orientation(rffgt_0,137,70). coords(rffgt_0,134,61,70).  visible(rffgt_0,70). happensAt(active(rffgt_0),70). orientation(rffgt_1,135,70). coords(rffgt_1,155,113,70).  visible(rffgt_1,70). happensAt(active(rffgt_1),70).
orientation(rffgt_0,134,71). coords(rffgt_0,134,61,71).  visible(rffgt_0,71). happensAt(active(rffgt_0),71). orientation(rffgt_1,134,71). coords(rffgt_1,155,113,71).  visible(rffgt_1,71). happensAt(active(rffgt_1),71).
orientation(rffgt_0,134,72). coords(rffgt_0,134,61,72).  visible(rffgt_0,72). happensAt(active(rffgt_0),72). orientation(rffgt_1,134,72). coords(rffgt_1,155,113,72).  visible(rffgt_1,72). happensAt(active(rffgt_1),72).
orientation(rffgt_0,134,73). coords(rffgt_0,134,61,73).  visible(rffgt_0,73). happensAt(active(rffgt_0),73). orientation(rffgt_1,134,73). coords(rffgt_1,155,114,73).  visible(rffgt_1,73). happensAt(active(rffgt_1),73).
orientation(rffgt_0,134,74). coords(rffgt_0,134,61,74).  visible(rffgt_0,74). happensAt(active(rffgt_0),74). orientation(rffgt_1,134,74). coords(rffgt_1,155,114,74).  visible(rffgt_1,74). happensAt(active(rffgt_1),74).
orientation(rffgt_0,134,75). coords(rffgt_0,133,60,75).  visible(rffgt_0,75). happensAt(active(rffgt_0),75). orientation(rffgt_1,134,75). coords(rffgt_1,156,114,75).  visible(rffgt_1,75). happensAt(active(rffgt_1),75).
orientation(rffgt_0,134,76). coords(rffgt_0,133,59,76).  visible(rffgt_0,76). happensAt(active(rffgt_0),76). orientation(rffgt_1,134,76). coords(rffgt_1,155,114,76).  visible(rffgt_1,76). happensAt(active(rffgt_1),76).
orientation(rffgt_0,134,77). coords(rffgt_0,133,59,77).  visible(rffgt_0,77). happensAt(active(rffgt_0),77). orientation(rffgt_1,138,77). coords(rffgt_1,155,114,77).  visible(rffgt_1,77). happensAt(active(rffgt_1),77).
orientation(rffgt_0,134,78). coords(rffgt_0,133,59,78).  visible(rffgt_0,78). happensAt(active(rffgt_0),78). orientation(rffgt_1,141,78). coords(rffgt_1,155,114,78).  visible(rffgt_1,78). happensAt(active(rffgt_1),78).
orientation(rffgt_0,134,79). coords(rffgt_0,133,58,79).  visible(rffgt_0,79). happensAt(active(rffgt_0),79). orientation(rffgt_1,143,79). coords(rffgt_1,155,114,79).  visible(rffgt_1,79). happensAt(active(rffgt_1),79).
orientation(rffgt_0,135,80). coords(rffgt_0,133,56,80).  visible(rffgt_0,80). happensAt(walking(rffgt_0),80). orientation(rffgt_1,143,80). coords(rffgt_1,156,114,80).  visible(rffgt_1,80). happensAt(active(rffgt_1),80).
orientation(rffgt_0,136,81). coords(rffgt_0,133,56,81).  visible(rffgt_0,81). happensAt(walking(rffgt_0),81). orientation(rffgt_1,143,81). coords(rffgt_1,156,114,81).  visible(rffgt_1,81). happensAt(active(rffgt_1),81).
orientation(rffgt_0,137,82). coords(rffgt_0,133,55,82).  visible(rffgt_0,82). happensAt(walking(rffgt_0),82). orientation(rffgt_1,143,82). coords(rffgt_1,156,113,82).  visible(rffgt_1,82). happensAt(active(rffgt_1),82).
orientation(rffgt_0,136,83). coords(rffgt_0,134,54,83).  visible(rffgt_0,83). happensAt(walking(rffgt_0),83). orientation(rffgt_1,143,83). coords(rffgt_1,156,111,83).  visible(rffgt_1,83). happensAt(active(rffgt_1),83).
orientation(rffgt_0,138,84). coords(rffgt_0,135,53,84).  visible(rffgt_0,84). happensAt(walking(rffgt_0),84). orientation(rffgt_1,143,84). coords(rffgt_1,156,109,84).  visible(rffgt_1,84). happensAt(active(rffgt_1),84).
#show holdsAt/2.