
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

	orientation(ricgt_3,40,63). coords(ricgt_3,265,276,63).  visible(ricgt_3,63). happensAt(walking(ricgt_3),63). orientation(ricgt_5,0,63). coords(ricgt_5,28,123,63).  visible(ricgt_5,63). happensAt(walking(ricgt_5),63).
orientation(ricgt_3,36,64). coords(ricgt_3,262,274,64).  visible(ricgt_3,64). happensAt(walking(ricgt_3),64). orientation(ricgt_5,0,64). coords(ricgt_5,28,123,64).  visible(ricgt_5,64). happensAt(walking(ricgt_5),64).
orientation(ricgt_3,32,65). coords(ricgt_3,258,272,65).  visible(ricgt_3,65). happensAt(walking(ricgt_3),65). orientation(ricgt_5,0,65). coords(ricgt_5,28,123,65).  visible(ricgt_5,65). happensAt(walking(ricgt_5),65).
orientation(ricgt_3,26,66). coords(ricgt_3,257,267,66).  visible(ricgt_3,66). happensAt(walking(ricgt_3),66). orientation(ricgt_5,0,66). coords(ricgt_5,28,123,66).  visible(ricgt_5,66). happensAt(walking(ricgt_5),66).
orientation(ricgt_3,23,67). coords(ricgt_3,257,263,67).  visible(ricgt_3,67). happensAt(walking(ricgt_3),67). orientation(ricgt_5,0,67). coords(ricgt_5,28,123,67).  visible(ricgt_5,67). happensAt(walking(ricgt_5),67).
orientation(ricgt_3,34,68). coords(ricgt_3,256,259,68).  visible(ricgt_3,68). happensAt(walking(ricgt_3),68). orientation(ricgt_5,0,68). coords(ricgt_5,28,123,68).  visible(ricgt_5,68). happensAt(walking(ricgt_5),68).
orientation(ricgt_3,27,69). coords(ricgt_3,256,255,69).  visible(ricgt_3,69). happensAt(walking(ricgt_3),69). orientation(ricgt_5,0,69). coords(ricgt_5,28,123,69).  visible(ricgt_5,69). happensAt(walking(ricgt_5),69).
orientation(ricgt_3,27,70). coords(ricgt_3,255,255,70).  visible(ricgt_3,70). happensAt(walking(ricgt_3),70). orientation(ricgt_5,0,70). coords(ricgt_5,28,123,70).  visible(ricgt_5,70). happensAt(walking(ricgt_5),70).
orientation(ricgt_3,27,71). coords(ricgt_3,254,254,71).  visible(ricgt_3,71). happensAt(walking(ricgt_3),71). orientation(ricgt_5,0,71). coords(ricgt_5,28,123,71).  visible(ricgt_5,71). happensAt(walking(ricgt_5),71).
orientation(ricgt_3,21,72). coords(ricgt_3,251,253,72).  visible(ricgt_3,72). happensAt(walking(ricgt_3),72). orientation(ricgt_5,0,72). coords(ricgt_5,28,123,72).  visible(ricgt_5,72). happensAt(walking(ricgt_5),72).
orientation(ricgt_3,17,73). coords(ricgt_3,249,251,73).  visible(ricgt_3,73). happensAt(walking(ricgt_3),73). orientation(ricgt_5,0,73). coords(ricgt_5,28,124,73).  visible(ricgt_5,73). happensAt(walking(ricgt_5),73).
orientation(ricgt_3,15,74). coords(ricgt_3,247,246,74).  visible(ricgt_3,74). happensAt(walking(ricgt_3),74). orientation(ricgt_5,0,74). coords(ricgt_5,28,124,74).  visible(ricgt_5,74). happensAt(walking(ricgt_5),74).
orientation(ricgt_3,10,75). coords(ricgt_3,246,242,75).  visible(ricgt_3,75). happensAt(walking(ricgt_3),75). orientation(ricgt_5,0,75). coords(ricgt_5,28,124,75).  visible(ricgt_5,75). happensAt(walking(ricgt_5),75).
orientation(ricgt_3,14,76). coords(ricgt_3,244,236,76).  visible(ricgt_3,76). happensAt(walking(ricgt_3),76). orientation(ricgt_5,0,76). coords(ricgt_5,28,124,76).  visible(ricgt_5,76). happensAt(walking(ricgt_5),76).
orientation(ricgt_3,12,77). coords(ricgt_3,242,233,77).  visible(ricgt_3,77). happensAt(walking(ricgt_3),77). orientation(ricgt_5,0,77). coords(ricgt_5,28,124,77).  visible(ricgt_5,77). happensAt(walking(ricgt_5),77).
orientation(ricgt_3,14,78). coords(ricgt_3,240,231,78).  visible(ricgt_3,78). happensAt(walking(ricgt_3),78). orientation(ricgt_5,0,78). coords(ricgt_5,28,124,78).  visible(ricgt_5,78). happensAt(walking(ricgt_5),78).
orientation(ricgt_3,11,79). coords(ricgt_3,239,229,79).  visible(ricgt_3,79). happensAt(walking(ricgt_3),79). orientation(ricgt_5,0,79). coords(ricgt_5,28,124,79).  visible(ricgt_5,79). happensAt(walking(ricgt_5),79).
orientation(ricgt_3,14,80). coords(ricgt_3,237,227,80).  visible(ricgt_3,80). happensAt(walking(ricgt_3),80). orientation(ricgt_5,0,80). coords(ricgt_5,28,124,80).  visible(ricgt_5,80). happensAt(walking(ricgt_5),80).
orientation(ricgt_3,12,81). coords(ricgt_3,237,225,81).  visible(ricgt_3,81). happensAt(walking(ricgt_3),81). orientation(ricgt_5,0,81). coords(ricgt_5,28,124,81).  visible(ricgt_5,81). happensAt(walking(ricgt_5),81).
orientation(ricgt_3,10,82). coords(ricgt_3,236,224,82).  visible(ricgt_3,82). happensAt(walking(ricgt_3),82). orientation(ricgt_5,0,82). coords(ricgt_5,28,124,82).  visible(ricgt_5,82). happensAt(walking(ricgt_5),82).
orientation(ricgt_3,8,83). coords(ricgt_3,235,223,83).  visible(ricgt_3,83). happensAt(walking(ricgt_3),83). orientation(ricgt_5,0,83). coords(ricgt_5,27,125,83).  visible(ricgt_5,83). happensAt(walking(ricgt_5),83).
orientation(ricgt_3,8,84). coords(ricgt_3,234,224,84).  visible(ricgt_3,84). happensAt(walking(ricgt_3),84). orientation(ricgt_5,0,84). coords(ricgt_5,27,127,84).  visible(ricgt_5,84). happensAt(walking(ricgt_5),84).
orientation(ricgt_3,6,85). coords(ricgt_3,233,221,85).  visible(ricgt_3,85). happensAt(walking(ricgt_3),85). orientation(ricgt_5,0,85). coords(ricgt_5,27,126,85).  visible(ricgt_5,85). happensAt(walking(ricgt_5),85).
orientation(ricgt_3,4,86). coords(ricgt_3,232,220,86).  visible(ricgt_3,86). happensAt(walking(ricgt_3),86). orientation(ricgt_5,0,86). coords(ricgt_5,27,126,86).  visible(ricgt_5,86). happensAt(walking(ricgt_5),86).
orientation(ricgt_3,2,87). coords(ricgt_3,231,219,87).  visible(ricgt_3,87). happensAt(walking(ricgt_3),87). orientation(ricgt_5,0,87). coords(ricgt_5,27,127,87).  visible(ricgt_5,87). happensAt(walking(ricgt_5),87).
#show holdsAt/2.