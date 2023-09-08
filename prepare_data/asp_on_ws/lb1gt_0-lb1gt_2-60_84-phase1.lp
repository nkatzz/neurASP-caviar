
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

	orientation(lb1gt_0,148,60). coords(lb1gt_0,181,119,60).  visible(lb1gt_0,60). happensAt(active(lb1gt_0),60). orientation(lb1gt_2,142,60). coords(lb1gt_2,83,79,60).  visible(lb1gt_2,60). happensAt(walking(lb1gt_2),60).
orientation(lb1gt_0,148,61). coords(lb1gt_0,180,119,61).  visible(lb1gt_0,61). happensAt(active(lb1gt_0),61). orientation(lb1gt_2,142,61). coords(lb1gt_2,83,80,61).  visible(lb1gt_2,61). happensAt(walking(lb1gt_2),61).
orientation(lb1gt_0,148,62). coords(lb1gt_0,180,119,62).  visible(lb1gt_0,62). happensAt(active(lb1gt_0),62). orientation(lb1gt_2,142,62). coords(lb1gt_2,83,80,62).  visible(lb1gt_2,62). happensAt(walking(lb1gt_2),62).
orientation(lb1gt_0,148,63). coords(lb1gt_0,179,119,63).  visible(lb1gt_0,63). happensAt(active(lb1gt_0),63). orientation(lb1gt_2,142,63). coords(lb1gt_2,83,80,63).  visible(lb1gt_2,63). happensAt(walking(lb1gt_2),63).
orientation(lb1gt_0,148,64). coords(lb1gt_0,178,119,64).  visible(lb1gt_0,64). happensAt(active(lb1gt_0),64). orientation(lb1gt_2,142,64). coords(lb1gt_2,83,80,64).  visible(lb1gt_2,64). happensAt(walking(lb1gt_2),64).
orientation(lb1gt_0,148,65). coords(lb1gt_0,177,118,65).  visible(lb1gt_0,65). happensAt(active(lb1gt_0),65). orientation(lb1gt_2,142,65). coords(lb1gt_2,84,80,65).  visible(lb1gt_2,65). happensAt(walking(lb1gt_2),65).
orientation(lb1gt_0,148,66). coords(lb1gt_0,176,118,66).  visible(lb1gt_0,66). happensAt(active(lb1gt_0),66). orientation(lb1gt_2,142,66). coords(lb1gt_2,84,81,66).  visible(lb1gt_2,66). happensAt(walking(lb1gt_2),66).
orientation(lb1gt_0,148,67). coords(lb1gt_0,176,117,67).  visible(lb1gt_0,67). happensAt(active(lb1gt_0),67). orientation(lb1gt_2,142,67). coords(lb1gt_2,84,81,67).  visible(lb1gt_2,67). happensAt(walking(lb1gt_2),67).
orientation(lb1gt_0,148,68). coords(lb1gt_0,176,118,68).  visible(lb1gt_0,68). happensAt(active(lb1gt_0),68). orientation(lb1gt_2,142,68). coords(lb1gt_2,85,81,68).  visible(lb1gt_2,68). happensAt(walking(lb1gt_2),68).
orientation(lb1gt_0,148,69). coords(lb1gt_0,176,117,69).  visible(lb1gt_0,69). happensAt(active(lb1gt_0),69). orientation(lb1gt_2,142,69). coords(lb1gt_2,85,81,69).  visible(lb1gt_2,69). happensAt(walking(lb1gt_2),69).
orientation(lb1gt_0,148,70). coords(lb1gt_0,175,116,70).  visible(lb1gt_0,70). happensAt(active(lb1gt_0),70). orientation(lb1gt_2,142,70). coords(lb1gt_2,86,81,70).  visible(lb1gt_2,70). happensAt(walking(lb1gt_2),70).
orientation(lb1gt_0,148,71). coords(lb1gt_0,175,115,71).  visible(lb1gt_0,71). happensAt(active(lb1gt_0),71). orientation(lb1gt_2,142,71). coords(lb1gt_2,86,81,71).  visible(lb1gt_2,71). happensAt(walking(lb1gt_2),71).
orientation(lb1gt_0,148,72). coords(lb1gt_0,175,113,72).  visible(lb1gt_0,72). happensAt(active(lb1gt_0),72). orientation(lb1gt_2,142,72). coords(lb1gt_2,87,81,72).  visible(lb1gt_2,72). happensAt(walking(lb1gt_2),72).
orientation(lb1gt_0,148,73). coords(lb1gt_0,173,112,73).  visible(lb1gt_0,73). happensAt(active(lb1gt_0),73). orientation(lb1gt_2,142,73). coords(lb1gt_2,87,81,73).  visible(lb1gt_2,73). happensAt(walking(lb1gt_2),73).
orientation(lb1gt_0,148,74). coords(lb1gt_0,172,112,74).  visible(lb1gt_0,74). happensAt(active(lb1gt_0),74). orientation(lb1gt_2,142,74). coords(lb1gt_2,87,81,74).  visible(lb1gt_2,74). happensAt(walking(lb1gt_2),74).
orientation(lb1gt_0,145,75). coords(lb1gt_0,170,111,75).  visible(lb1gt_0,75). happensAt(active(lb1gt_0),75). orientation(lb1gt_2,142,75). coords(lb1gt_2,88,81,75).  visible(lb1gt_2,75). happensAt(walking(lb1gt_2),75).
orientation(lb1gt_0,145,76). coords(lb1gt_0,169,110,76).  visible(lb1gt_0,76). happensAt(active(lb1gt_0),76). orientation(lb1gt_2,142,76). coords(lb1gt_2,88,81,76).  visible(lb1gt_2,76). happensAt(walking(lb1gt_2),76).
orientation(lb1gt_0,145,77). coords(lb1gt_0,168,110,77).  visible(lb1gt_0,77). happensAt(active(lb1gt_0),77). orientation(lb1gt_2,142,77). coords(lb1gt_2,88,81,77).  visible(lb1gt_2,77). happensAt(walking(lb1gt_2),77).
orientation(lb1gt_0,145,78). coords(lb1gt_0,167,110,78).  visible(lb1gt_0,78). happensAt(active(lb1gt_0),78). orientation(lb1gt_2,142,78). coords(lb1gt_2,88,81,78).  visible(lb1gt_2,78). happensAt(walking(lb1gt_2),78).
orientation(lb1gt_0,145,79). coords(lb1gt_0,167,109,79).  visible(lb1gt_0,79). happensAt(active(lb1gt_0),79). orientation(lb1gt_2,142,79). coords(lb1gt_2,88,81,79).  visible(lb1gt_2,79). happensAt(walking(lb1gt_2),79).
orientation(lb1gt_0,145,80). coords(lb1gt_0,167,109,80).  visible(lb1gt_0,80). happensAt(active(lb1gt_0),80). orientation(lb1gt_2,142,80). coords(lb1gt_2,88,82,80).  visible(lb1gt_2,80). happensAt(walking(lb1gt_2),80).
orientation(lb1gt_0,145,81). coords(lb1gt_0,167,109,81).  visible(lb1gt_0,81). happensAt(active(lb1gt_0),81). orientation(lb1gt_2,142,81). coords(lb1gt_2,88,82,81).  visible(lb1gt_2,81). happensAt(walking(lb1gt_2),81).
orientation(lb1gt_0,145,82). coords(lb1gt_0,166,109,82).  visible(lb1gt_0,82). happensAt(active(lb1gt_0),82). orientation(lb1gt_2,142,82). coords(lb1gt_2,88,83,82).  visible(lb1gt_2,82). happensAt(walking(lb1gt_2),82).
orientation(lb1gt_0,145,83). coords(lb1gt_0,166,108,83).  visible(lb1gt_0,83). happensAt(active(lb1gt_0),83). orientation(lb1gt_2,142,83). coords(lb1gt_2,88,83,83).  visible(lb1gt_2,83). happensAt(walking(lb1gt_2),83).
orientation(lb1gt_0,145,84). coords(lb1gt_0,166,108,84).  visible(lb1gt_0,84). happensAt(active(lb1gt_0),84). orientation(lb1gt_2,142,84). coords(lb1gt_2,88,84,84).  visible(lb1gt_2,84). happensAt(walking(lb1gt_2),84).
#show holdsAt/2.