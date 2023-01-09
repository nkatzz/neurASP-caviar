
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

	orientation(fcgt_0,143,60). coords(fcgt_0,198,150,60).  visible(fcgt_0,60). happensAt(walking(fcgt_0),60). orientation(fcgt_4,165,60). coords(fcgt_4,100,208,60).  visible(fcgt_4,60). happensAt(inactive(fcgt_4),60).
orientation(fcgt_0,143,61). coords(fcgt_0,198,150,61).  visible(fcgt_0,61). happensAt(walking(fcgt_0),61). orientation(fcgt_4,165,61). coords(fcgt_4,100,208,61).  visible(fcgt_4,61). happensAt(inactive(fcgt_4),61).
orientation(fcgt_0,143,62). coords(fcgt_0,199,151,62).  visible(fcgt_0,62). happensAt(walking(fcgt_0),62). orientation(fcgt_4,165,62). coords(fcgt_4,100,208,62).  visible(fcgt_4,62). happensAt(inactive(fcgt_4),62).
orientation(fcgt_0,143,63). coords(fcgt_0,199,151,63).  visible(fcgt_0,63). happensAt(walking(fcgt_0),63). orientation(fcgt_4,165,63). coords(fcgt_4,100,208,63).  visible(fcgt_4,63). happensAt(inactive(fcgt_4),63).
orientation(fcgt_0,143,64). coords(fcgt_0,199,151,64).  visible(fcgt_0,64). happensAt(walking(fcgt_0),64). orientation(fcgt_4,165,64). coords(fcgt_4,100,208,64).  visible(fcgt_4,64). happensAt(inactive(fcgt_4),64).
orientation(fcgt_0,143,65). coords(fcgt_0,199,151,65).  visible(fcgt_0,65). happensAt(inactive(fcgt_0),65). orientation(fcgt_4,165,65). coords(fcgt_4,100,208,65).  visible(fcgt_4,65). happensAt(inactive(fcgt_4),65).
orientation(fcgt_0,143,66). coords(fcgt_0,199,151,66).  visible(fcgt_0,66). happensAt(inactive(fcgt_0),66). orientation(fcgt_4,165,66). coords(fcgt_4,100,208,66).  visible(fcgt_4,66). happensAt(inactive(fcgt_4),66).
orientation(fcgt_0,143,67). coords(fcgt_0,199,151,67).  visible(fcgt_0,67). happensAt(inactive(fcgt_0),67). orientation(fcgt_4,165,67). coords(fcgt_4,100,208,67).  visible(fcgt_4,67). happensAt(inactive(fcgt_4),67).
orientation(fcgt_0,143,68). coords(fcgt_0,199,151,68).  visible(fcgt_0,68). happensAt(inactive(fcgt_0),68). orientation(fcgt_4,165,68). coords(fcgt_4,100,208,68).  visible(fcgt_4,68). happensAt(inactive(fcgt_4),68).
orientation(fcgt_0,143,69). coords(fcgt_0,199,151,69).  visible(fcgt_0,69). happensAt(inactive(fcgt_0),69). orientation(fcgt_4,165,69). coords(fcgt_4,100,208,69).  visible(fcgt_4,69). happensAt(inactive(fcgt_4),69).
orientation(fcgt_0,143,70). coords(fcgt_0,199,151,70).  visible(fcgt_0,70). happensAt(inactive(fcgt_0),70). orientation(fcgt_4,165,70). coords(fcgt_4,100,208,70).  visible(fcgt_4,70). happensAt(inactive(fcgt_4),70).
orientation(fcgt_0,143,71). coords(fcgt_0,199,151,71).  visible(fcgt_0,71). happensAt(inactive(fcgt_0),71). orientation(fcgt_4,165,71). coords(fcgt_4,100,208,71).  visible(fcgt_4,71). happensAt(inactive(fcgt_4),71).
orientation(fcgt_0,143,72). coords(fcgt_0,199,151,72).  visible(fcgt_0,72). happensAt(inactive(fcgt_0),72). orientation(fcgt_4,165,72). coords(fcgt_4,100,208,72).  visible(fcgt_4,72). happensAt(inactive(fcgt_4),72).
orientation(fcgt_0,143,73). coords(fcgt_0,199,151,73).  visible(fcgt_0,73). happensAt(inactive(fcgt_0),73). orientation(fcgt_4,165,73). coords(fcgt_4,100,208,73).  visible(fcgt_4,73). happensAt(inactive(fcgt_4),73).
orientation(fcgt_0,143,74). coords(fcgt_0,199,151,74).  visible(fcgt_0,74). happensAt(inactive(fcgt_0),74). orientation(fcgt_4,165,74). coords(fcgt_4,100,208,74).  visible(fcgt_4,74). happensAt(inactive(fcgt_4),74).
orientation(fcgt_0,143,75). coords(fcgt_0,200,151,75).  visible(fcgt_0,75). happensAt(inactive(fcgt_0),75). orientation(fcgt_4,165,75). coords(fcgt_4,100,208,75).  visible(fcgt_4,75). happensAt(inactive(fcgt_4),75).
orientation(fcgt_0,143,76). coords(fcgt_0,200,151,76).  visible(fcgt_0,76). happensAt(inactive(fcgt_0),76). orientation(fcgt_4,165,76). coords(fcgt_4,100,208,76).  visible(fcgt_4,76). happensAt(inactive(fcgt_4),76).
orientation(fcgt_0,143,77). coords(fcgt_0,200,151,77).  visible(fcgt_0,77). happensAt(inactive(fcgt_0),77). orientation(fcgt_4,165,77). coords(fcgt_4,100,208,77).  visible(fcgt_4,77). happensAt(inactive(fcgt_4),77).
orientation(fcgt_0,143,78). coords(fcgt_0,200,151,78).  visible(fcgt_0,78). happensAt(inactive(fcgt_0),78). orientation(fcgt_4,165,78). coords(fcgt_4,100,209,78).  visible(fcgt_4,78). happensAt(inactive(fcgt_4),78).
orientation(fcgt_0,143,79). coords(fcgt_0,200,151,79).  visible(fcgt_0,79). happensAt(inactive(fcgt_0),79). orientation(fcgt_4,165,79). coords(fcgt_4,100,209,79).  visible(fcgt_4,79). happensAt(inactive(fcgt_4),79).
orientation(fcgt_0,143,80). coords(fcgt_0,200,152,80).  visible(fcgt_0,80). happensAt(inactive(fcgt_0),80). orientation(fcgt_4,165,80). coords(fcgt_4,100,209,80).  visible(fcgt_4,80). happensAt(inactive(fcgt_4),80).
orientation(fcgt_0,143,81). coords(fcgt_0,200,152,81).  visible(fcgt_0,81). happensAt(inactive(fcgt_0),81). orientation(fcgt_4,165,81). coords(fcgt_4,100,209,81).  visible(fcgt_4,81). happensAt(inactive(fcgt_4),81).
orientation(fcgt_0,143,82). coords(fcgt_0,201,153,82).  visible(fcgt_0,82). happensAt(inactive(fcgt_0),82). orientation(fcgt_4,165,82). coords(fcgt_4,100,209,82).  visible(fcgt_4,82). happensAt(inactive(fcgt_4),82).
orientation(fcgt_0,147,83). coords(fcgt_0,201,153,83).  visible(fcgt_0,83). happensAt(walking(fcgt_0),83). orientation(fcgt_4,165,83). coords(fcgt_4,100,209,83).  visible(fcgt_4,83). happensAt(inactive(fcgt_4),83).
orientation(fcgt_0,147,84). coords(fcgt_0,202,154,84).  visible(fcgt_0,84). happensAt(walking(fcgt_0),84). orientation(fcgt_4,165,84). coords(fcgt_4,100,209,84).  visible(fcgt_4,84). happensAt(inactive(fcgt_4),84).
#show holdsAt/2.