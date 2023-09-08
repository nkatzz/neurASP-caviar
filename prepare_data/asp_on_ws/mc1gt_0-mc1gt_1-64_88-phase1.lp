
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

	orientation(mc1gt_0,148,64). coords(mc1gt_0,75,81,64).  visible(mc1gt_0,64). happensAt(walking(mc1gt_0),64). orientation(mc1gt_1,145,64). coords(mc1gt_1,82,77,64).  visible(mc1gt_1,64). happensAt(walking(mc1gt_1),64).
orientation(mc1gt_0,148,65). coords(mc1gt_0,74,81,65).  visible(mc1gt_0,65). happensAt(walking(mc1gt_0),65). orientation(mc1gt_1,145,65). coords(mc1gt_1,82,78,65).  visible(mc1gt_1,65). happensAt(walking(mc1gt_1),65).
orientation(mc1gt_0,148,66). coords(mc1gt_0,75,81,66).  visible(mc1gt_0,66). happensAt(walking(mc1gt_0),66). orientation(mc1gt_1,145,66). coords(mc1gt_1,82,78,66).  visible(mc1gt_1,66). happensAt(walking(mc1gt_1),66).
orientation(mc1gt_0,148,67). coords(mc1gt_0,75,81,67).  visible(mc1gt_0,67). happensAt(walking(mc1gt_0),67). orientation(mc1gt_1,145,67). coords(mc1gt_1,83,79,67).  visible(mc1gt_1,67). happensAt(walking(mc1gt_1),67).
orientation(mc1gt_0,148,68). coords(mc1gt_0,75,82,68).  visible(mc1gt_0,68). happensAt(walking(mc1gt_0),68). orientation(mc1gt_1,145,68). coords(mc1gt_1,83,79,68).  visible(mc1gt_1,68). happensAt(walking(mc1gt_1),68).
orientation(mc1gt_0,148,69). coords(mc1gt_0,79,96,69).  visible(mc1gt_0,69). happensAt(walking(mc1gt_0),69). orientation(mc1gt_1,145,69). coords(mc1gt_1,94,83,69).  visible(mc1gt_1,69). happensAt(walking(mc1gt_1),69).
orientation(mc1gt_0,148,70). coords(mc1gt_0,79,96,70).  visible(mc1gt_0,70). happensAt(walking(mc1gt_0),70). orientation(mc1gt_1,145,70). coords(mc1gt_1,94,83,70).  visible(mc1gt_1,70). happensAt(walking(mc1gt_1),70).
orientation(mc1gt_0,148,71). coords(mc1gt_0,79,96,71).  visible(mc1gt_0,71). happensAt(walking(mc1gt_0),71). orientation(mc1gt_1,145,71). coords(mc1gt_1,94,83,71).  visible(mc1gt_1,71). happensAt(walking(mc1gt_1),71).
orientation(mc1gt_0,148,72). coords(mc1gt_0,79,96,72).  visible(mc1gt_0,72). happensAt(walking(mc1gt_0),72). orientation(mc1gt_1,145,72). coords(mc1gt_1,94,83,72).  visible(mc1gt_1,72). happensAt(walking(mc1gt_1),72).
orientation(mc1gt_0,148,73). coords(mc1gt_0,79,96,73).  visible(mc1gt_0,73). happensAt(walking(mc1gt_0),73). orientation(mc1gt_1,145,73). coords(mc1gt_1,94,83,73).  visible(mc1gt_1,73). happensAt(walking(mc1gt_1),73).
orientation(mc1gt_0,148,74). coords(mc1gt_0,79,96,74).  visible(mc1gt_0,74). happensAt(walking(mc1gt_0),74). orientation(mc1gt_1,145,74). coords(mc1gt_1,94,83,74).  visible(mc1gt_1,74). happensAt(walking(mc1gt_1),74).
orientation(mc1gt_0,148,75). coords(mc1gt_0,79,96,75).  visible(mc1gt_0,75). happensAt(walking(mc1gt_0),75). orientation(mc1gt_1,145,75). coords(mc1gt_1,94,83,75).  visible(mc1gt_1,75). happensAt(walking(mc1gt_1),75).
orientation(mc1gt_0,148,76). coords(mc1gt_0,79,96,76).  visible(mc1gt_0,76). happensAt(walking(mc1gt_0),76). orientation(mc1gt_1,145,76). coords(mc1gt_1,94,83,76).  visible(mc1gt_1,76). happensAt(walking(mc1gt_1),76).
orientation(mc1gt_0,148,77). coords(mc1gt_0,79,96,77).  visible(mc1gt_0,77). happensAt(walking(mc1gt_0),77). orientation(mc1gt_1,145,77). coords(mc1gt_1,94,83,77).  visible(mc1gt_1,77). happensAt(walking(mc1gt_1),77).
orientation(mc1gt_0,148,78). coords(mc1gt_0,79,96,78).  visible(mc1gt_0,78). happensAt(walking(mc1gt_0),78). orientation(mc1gt_1,145,78). coords(mc1gt_1,94,83,78).  visible(mc1gt_1,78). happensAt(walking(mc1gt_1),78).
orientation(mc1gt_0,148,79). coords(mc1gt_0,79,96,79).  visible(mc1gt_0,79). happensAt(walking(mc1gt_0),79). orientation(mc1gt_1,145,79). coords(mc1gt_1,94,83,79).  visible(mc1gt_1,79). happensAt(walking(mc1gt_1),79).
orientation(mc1gt_0,148,80). coords(mc1gt_0,79,96,80).  visible(mc1gt_0,80). happensAt(walking(mc1gt_0),80). orientation(mc1gt_1,145,80). coords(mc1gt_1,94,83,80).  visible(mc1gt_1,80). happensAt(walking(mc1gt_1),80).
orientation(mc1gt_0,148,81). coords(mc1gt_0,79,96,81).  visible(mc1gt_0,81). happensAt(walking(mc1gt_0),81). orientation(mc1gt_1,145,81). coords(mc1gt_1,94,83,81).  visible(mc1gt_1,81). happensAt(walking(mc1gt_1),81).
orientation(mc1gt_0,148,82). coords(mc1gt_0,79,96,82).  visible(mc1gt_0,82). happensAt(walking(mc1gt_0),82). orientation(mc1gt_1,145,82). coords(mc1gt_1,94,83,82).  visible(mc1gt_1,82). happensAt(walking(mc1gt_1),82).
orientation(mc1gt_0,148,83). coords(mc1gt_0,79,96,83).  visible(mc1gt_0,83). happensAt(walking(mc1gt_0),83). orientation(mc1gt_1,145,83). coords(mc1gt_1,94,83,83).  visible(mc1gt_1,83). happensAt(walking(mc1gt_1),83).
orientation(mc1gt_0,148,84). coords(mc1gt_0,79,96,84).  visible(mc1gt_0,84). happensAt(walking(mc1gt_0),84). orientation(mc1gt_1,145,84). coords(mc1gt_1,94,83,84).  visible(mc1gt_1,84). happensAt(walking(mc1gt_1),84).
orientation(mc1gt_0,148,85). coords(mc1gt_0,79,96,85).  visible(mc1gt_0,85). happensAt(walking(mc1gt_0),85). orientation(mc1gt_1,145,85). coords(mc1gt_1,94,83,85).  visible(mc1gt_1,85). happensAt(walking(mc1gt_1),85).
orientation(mc1gt_0,148,86). coords(mc1gt_0,79,96,86).  visible(mc1gt_0,86). happensAt(walking(mc1gt_0),86). orientation(mc1gt_1,145,86). coords(mc1gt_1,94,83,86).  visible(mc1gt_1,86). happensAt(walking(mc1gt_1),86).
orientation(mc1gt_0,148,87). coords(mc1gt_0,79,96,87).  visible(mc1gt_0,87). happensAt(walking(mc1gt_0),87). orientation(mc1gt_1,145,87). coords(mc1gt_1,94,83,87).  visible(mc1gt_1,87). happensAt(walking(mc1gt_1),87).
orientation(mc1gt_0,148,88). coords(mc1gt_0,79,96,88).  visible(mc1gt_0,88). happensAt(walking(mc1gt_0),88). orientation(mc1gt_1,145,88). coords(mc1gt_1,94,83,88).  visible(mc1gt_1,88). happensAt(walking(mc1gt_1),88).
#show holdsAt/2.