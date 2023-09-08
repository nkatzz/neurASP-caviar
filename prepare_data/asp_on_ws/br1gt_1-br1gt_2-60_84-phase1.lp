
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

	orientation(br1gt_1,164,60). coords(br1gt_1,109,185,60).  visible(br1gt_1,60). happensAt(walking(br1gt_1),60). orientation(br1gt_2,171,60). coords(br1gt_2,86,207,60).  visible(br1gt_2,60). happensAt(inactive(br1gt_2),60).
orientation(br1gt_1,164,61). coords(br1gt_1,108,184,61).  visible(br1gt_1,61). happensAt(walking(br1gt_1),61). orientation(br1gt_2,171,61). coords(br1gt_2,86,207,61).  visible(br1gt_2,61). happensAt(inactive(br1gt_2),61).
orientation(br1gt_1,164,62). coords(br1gt_1,108,184,62).  visible(br1gt_1,62). happensAt(walking(br1gt_1),62). orientation(br1gt_2,171,62). coords(br1gt_2,86,207,62).  visible(br1gt_2,62). happensAt(inactive(br1gt_2),62).
orientation(br1gt_1,164,63). coords(br1gt_1,107,183,63).  visible(br1gt_1,63). happensAt(walking(br1gt_1),63). orientation(br1gt_2,171,63). coords(br1gt_2,86,207,63).  visible(br1gt_2,63). happensAt(inactive(br1gt_2),63).
orientation(br1gt_1,164,64). coords(br1gt_1,107,183,64).  visible(br1gt_1,64). happensAt(walking(br1gt_1),64). orientation(br1gt_2,171,64). coords(br1gt_2,86,207,64).  visible(br1gt_2,64). happensAt(inactive(br1gt_2),64).
orientation(br1gt_1,164,65). coords(br1gt_1,107,183,65).  visible(br1gt_1,65). happensAt(walking(br1gt_1),65). orientation(br1gt_2,171,65). coords(br1gt_2,86,207,65).  visible(br1gt_2,65). happensAt(inactive(br1gt_2),65).
orientation(br1gt_1,164,66). coords(br1gt_1,106,183,66).  visible(br1gt_1,66). happensAt(walking(br1gt_1),66). orientation(br1gt_2,171,66). coords(br1gt_2,86,207,66).  visible(br1gt_2,66). happensAt(inactive(br1gt_2),66).
orientation(br1gt_1,164,67). coords(br1gt_1,105,183,67).  visible(br1gt_1,67). happensAt(walking(br1gt_1),67). orientation(br1gt_2,171,67). coords(br1gt_2,86,207,67).  visible(br1gt_2,67). happensAt(inactive(br1gt_2),67).
orientation(br1gt_1,164,68). coords(br1gt_1,104,183,68).  visible(br1gt_1,68). happensAt(walking(br1gt_1),68). orientation(br1gt_2,171,68). coords(br1gt_2,86,207,68).  visible(br1gt_2,68). happensAt(inactive(br1gt_2),68).
orientation(br1gt_1,164,69). coords(br1gt_1,102,181,69).  visible(br1gt_1,69). happensAt(walking(br1gt_1),69). orientation(br1gt_2,171,69). coords(br1gt_2,86,207,69).  visible(br1gt_2,69). happensAt(inactive(br1gt_2),69).
orientation(br1gt_1,164,70). coords(br1gt_1,100,180,70).  visible(br1gt_1,70). happensAt(walking(br1gt_1),70). orientation(br1gt_2,171,70). coords(br1gt_2,86,207,70).  visible(br1gt_2,70). happensAt(inactive(br1gt_2),70).
orientation(br1gt_1,164,71). coords(br1gt_1,99,179,71).  visible(br1gt_1,71). happensAt(walking(br1gt_1),71). orientation(br1gt_2,171,71). coords(br1gt_2,86,207,71).  visible(br1gt_2,71). happensAt(inactive(br1gt_2),71).
orientation(br1gt_1,164,72). coords(br1gt_1,97,178,72).  visible(br1gt_1,72). happensAt(walking(br1gt_1),72). orientation(br1gt_2,171,72). coords(br1gt_2,86,207,72).  visible(br1gt_2,72). happensAt(inactive(br1gt_2),72).
orientation(br1gt_1,164,73). coords(br1gt_1,96,177,73).  visible(br1gt_1,73). happensAt(walking(br1gt_1),73). orientation(br1gt_2,171,73). coords(br1gt_2,86,207,73).  visible(br1gt_2,73). happensAt(inactive(br1gt_2),73).
orientation(br1gt_1,164,74). coords(br1gt_1,95,176,74).  visible(br1gt_1,74). happensAt(walking(br1gt_1),74). orientation(br1gt_2,171,74). coords(br1gt_2,86,207,74).  visible(br1gt_2,74). happensAt(inactive(br1gt_2),74).
orientation(br1gt_1,164,75). coords(br1gt_1,95,176,75).  visible(br1gt_1,75). happensAt(walking(br1gt_1),75). orientation(br1gt_2,171,75). coords(br1gt_2,86,207,75).  visible(br1gt_2,75). happensAt(inactive(br1gt_2),75).
orientation(br1gt_1,164,76). coords(br1gt_1,95,176,76).  visible(br1gt_1,76). happensAt(walking(br1gt_1),76). orientation(br1gt_2,171,76). coords(br1gt_2,86,207,76).  visible(br1gt_2,76). happensAt(inactive(br1gt_2),76).
orientation(br1gt_1,164,77). coords(br1gt_1,94,176,77).  visible(br1gt_1,77). happensAt(walking(br1gt_1),77). orientation(br1gt_2,171,77). coords(br1gt_2,86,207,77).  visible(br1gt_2,77). happensAt(inactive(br1gt_2),77).
orientation(br1gt_1,164,78). coords(br1gt_1,94,176,78).  visible(br1gt_1,78). happensAt(walking(br1gt_1),78). orientation(br1gt_2,171,78). coords(br1gt_2,86,207,78).  visible(br1gt_2,78). happensAt(inactive(br1gt_2),78).
orientation(br1gt_1,164,79). coords(br1gt_1,93,175,79).  visible(br1gt_1,79). happensAt(walking(br1gt_1),79). orientation(br1gt_2,171,79). coords(br1gt_2,86,207,79).  visible(br1gt_2,79). happensAt(inactive(br1gt_2),79).
orientation(br1gt_1,164,80). coords(br1gt_1,93,175,80).  visible(br1gt_1,80). happensAt(walking(br1gt_1),80). orientation(br1gt_2,171,80). coords(br1gt_2,86,207,80).  visible(br1gt_2,80). happensAt(inactive(br1gt_2),80).
orientation(br1gt_1,164,81). coords(br1gt_1,92,173,81).  visible(br1gt_1,81). happensAt(walking(br1gt_1),81). orientation(br1gt_2,171,81). coords(br1gt_2,86,207,81).  visible(br1gt_2,81). happensAt(inactive(br1gt_2),81).
orientation(br1gt_1,164,82). coords(br1gt_1,90,171,82).  visible(br1gt_1,82). happensAt(walking(br1gt_1),82). orientation(br1gt_2,171,82). coords(br1gt_2,86,207,82).  visible(br1gt_2,82). happensAt(inactive(br1gt_2),82).
orientation(br1gt_1,164,83). coords(br1gt_1,88,171,83).  visible(br1gt_1,83). happensAt(walking(br1gt_1),83). orientation(br1gt_2,171,83). coords(br1gt_2,86,207,83).  visible(br1gt_2,83). happensAt(inactive(br1gt_2),83).
orientation(br1gt_1,164,84). coords(br1gt_1,87,171,84).  visible(br1gt_1,84). happensAt(walking(br1gt_1),84). orientation(br1gt_2,171,84). coords(br1gt_2,86,207,84).  visible(br1gt_2,84). happensAt(inactive(br1gt_2),84).
#show holdsAt/2.