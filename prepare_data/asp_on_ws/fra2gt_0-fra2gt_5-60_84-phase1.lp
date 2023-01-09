
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

	orientation(fra2gt_0,150,60). coords(fra2gt_0,157,129,60).  visible(fra2gt_0,60). happensAt(inactive(fra2gt_0),60). orientation(fra2gt_5,150,60). coords(fra2gt_5,35,110,60).  visible(fra2gt_5,60). happensAt(walking(fra2gt_5),60).
orientation(fra2gt_0,150,61). coords(fra2gt_0,157,129,61).  visible(fra2gt_0,61). happensAt(inactive(fra2gt_0),61). orientation(fra2gt_5,150,61). coords(fra2gt_5,35,110,61).  visible(fra2gt_5,61). happensAt(walking(fra2gt_5),61).
orientation(fra2gt_0,150,62). coords(fra2gt_0,157,129,62).  visible(fra2gt_0,62). happensAt(inactive(fra2gt_0),62). orientation(fra2gt_5,150,62). coords(fra2gt_5,35,110,62).  visible(fra2gt_5,62). happensAt(walking(fra2gt_5),62).
orientation(fra2gt_0,150,63). coords(fra2gt_0,157,129,63).  visible(fra2gt_0,63). happensAt(inactive(fra2gt_0),63). orientation(fra2gt_5,150,63). coords(fra2gt_5,35,110,63).  visible(fra2gt_5,63). happensAt(walking(fra2gt_5),63).
orientation(fra2gt_0,150,64). coords(fra2gt_0,157,129,64).  visible(fra2gt_0,64). happensAt(inactive(fra2gt_0),64). orientation(fra2gt_5,150,64). coords(fra2gt_5,35,110,64).  visible(fra2gt_5,64). happensAt(walking(fra2gt_5),64).
orientation(fra2gt_0,150,65). coords(fra2gt_0,157,129,65).  visible(fra2gt_0,65). happensAt(inactive(fra2gt_0),65). orientation(fra2gt_5,150,65). coords(fra2gt_5,34,110,65).  visible(fra2gt_5,65). happensAt(walking(fra2gt_5),65).
orientation(fra2gt_0,150,66). coords(fra2gt_0,157,129,66).  visible(fra2gt_0,66). happensAt(inactive(fra2gt_0),66). orientation(fra2gt_5,150,66). coords(fra2gt_5,34,110,66).  visible(fra2gt_5,66). happensAt(walking(fra2gt_5),66).
orientation(fra2gt_0,150,67). coords(fra2gt_0,157,129,67).  visible(fra2gt_0,67). happensAt(inactive(fra2gt_0),67). orientation(fra2gt_5,150,67). coords(fra2gt_5,34,110,67).  visible(fra2gt_5,67). happensAt(walking(fra2gt_5),67).
orientation(fra2gt_0,150,68). coords(fra2gt_0,157,129,68).  visible(fra2gt_0,68). happensAt(inactive(fra2gt_0),68). orientation(fra2gt_5,150,68). coords(fra2gt_5,34,110,68).  visible(fra2gt_5,68). happensAt(walking(fra2gt_5),68).
orientation(fra2gt_0,150,69). coords(fra2gt_0,157,129,69).  visible(fra2gt_0,69). happensAt(active(fra2gt_0),69). orientation(fra2gt_5,150,69). coords(fra2gt_5,32,110,69).  visible(fra2gt_5,69). happensAt(walking(fra2gt_5),69).
orientation(fra2gt_0,150,70). coords(fra2gt_0,157,129,70).  visible(fra2gt_0,70). happensAt(active(fra2gt_0),70). orientation(fra2gt_5,150,70). coords(fra2gt_5,32,110,70).  visible(fra2gt_5,70). happensAt(walking(fra2gt_5),70).
orientation(fra2gt_0,150,71). coords(fra2gt_0,157,129,71).  visible(fra2gt_0,71). happensAt(active(fra2gt_0),71). orientation(fra2gt_5,150,71). coords(fra2gt_5,32,110,71).  visible(fra2gt_5,71). happensAt(walking(fra2gt_5),71).
orientation(fra2gt_0,150,72). coords(fra2gt_0,157,129,72).  visible(fra2gt_0,72). happensAt(active(fra2gt_0),72). orientation(fra2gt_5,150,72). coords(fra2gt_5,32,112,72).  visible(fra2gt_5,72). happensAt(walking(fra2gt_5),72).
orientation(fra2gt_0,150,73). coords(fra2gt_0,157,129,73).  visible(fra2gt_0,73). happensAt(active(fra2gt_0),73). orientation(fra2gt_5,150,73). coords(fra2gt_5,32,113,73).  visible(fra2gt_5,73). happensAt(walking(fra2gt_5),73).
orientation(fra2gt_0,150,74). coords(fra2gt_0,157,129,74).  visible(fra2gt_0,74). happensAt(active(fra2gt_0),74). orientation(fra2gt_5,150,74). coords(fra2gt_5,32,113,74).  visible(fra2gt_5,74). happensAt(walking(fra2gt_5),74).
orientation(fra2gt_0,150,75). coords(fra2gt_0,158,129,75).  visible(fra2gt_0,75). happensAt(active(fra2gt_0),75). orientation(fra2gt_5,150,75). coords(fra2gt_5,32,113,75).  visible(fra2gt_5,75). happensAt(walking(fra2gt_5),75).
orientation(fra2gt_0,150,76). coords(fra2gt_0,158,129,76).  visible(fra2gt_0,76). happensAt(active(fra2gt_0),76). orientation(fra2gt_5,150,76). coords(fra2gt_5,32,113,76).  visible(fra2gt_5,76). happensAt(walking(fra2gt_5),76).
orientation(fra2gt_0,150,77). coords(fra2gt_0,160,130,77).  visible(fra2gt_0,77). happensAt(active(fra2gt_0),77). orientation(fra2gt_5,150,77). coords(fra2gt_5,32,115,77).  visible(fra2gt_5,77). happensAt(walking(fra2gt_5),77).
orientation(fra2gt_0,150,78). coords(fra2gt_0,161,130,78).  visible(fra2gt_0,78). happensAt(active(fra2gt_0),78). orientation(fra2gt_5,150,78). coords(fra2gt_5,32,115,78).  visible(fra2gt_5,78). happensAt(walking(fra2gt_5),78).
orientation(fra2gt_0,150,79). coords(fra2gt_0,163,130,79).  visible(fra2gt_0,79). happensAt(active(fra2gt_0),79). orientation(fra2gt_5,150,79). coords(fra2gt_5,32,115,79).  visible(fra2gt_5,79). happensAt(walking(fra2gt_5),79).
orientation(fra2gt_0,150,80). coords(fra2gt_0,165,131,80).  visible(fra2gt_0,80). happensAt(active(fra2gt_0),80). orientation(fra2gt_5,150,80). coords(fra2gt_5,32,116,80).  visible(fra2gt_5,80). happensAt(walking(fra2gt_5),80).
orientation(fra2gt_0,150,81). coords(fra2gt_0,166,131,81).  visible(fra2gt_0,81). happensAt(active(fra2gt_0),81). orientation(fra2gt_5,150,81). coords(fra2gt_5,32,116,81).  visible(fra2gt_5,81). happensAt(walking(fra2gt_5),81).
orientation(fra2gt_0,150,82). coords(fra2gt_0,168,131,82).  visible(fra2gt_0,82). happensAt(walking(fra2gt_0),82). orientation(fra2gt_5,150,82). coords(fra2gt_5,32,116,82).  visible(fra2gt_5,82). happensAt(walking(fra2gt_5),82).
orientation(fra2gt_0,150,83). coords(fra2gt_0,170,132,83).  visible(fra2gt_0,83). happensAt(walking(fra2gt_0),83). orientation(fra2gt_5,150,83). coords(fra2gt_5,32,116,83).  visible(fra2gt_5,83). happensAt(walking(fra2gt_5),83).
orientation(fra2gt_0,150,84). coords(fra2gt_0,173,132,84).  visible(fra2gt_0,84). happensAt(walking(fra2gt_0),84). orientation(fra2gt_5,150,84). coords(fra2gt_5,32,116,84).  visible(fra2gt_5,84). happensAt(walking(fra2gt_5),84).
#show holdsAt/2.