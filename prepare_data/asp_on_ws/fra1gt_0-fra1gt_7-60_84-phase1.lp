
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

	orientation(fra1gt_0,5,60). coords(fra1gt_0,247,227,60).  visible(fra1gt_0,60). happensAt(walking(fra1gt_0),60). orientation(fra1gt_7,136,60). coords(fra1gt_7,144,41,60).  visible(fra1gt_7,60). happensAt(inactive(fra1gt_7),60).
orientation(fra1gt_0,17,61). coords(fra1gt_0,250,230,61).  visible(fra1gt_0,61). happensAt(walking(fra1gt_0),61). orientation(fra1gt_7,136,61). coords(fra1gt_7,144,41,61).  visible(fra1gt_7,61). happensAt(inactive(fra1gt_7),61).
orientation(fra1gt_0,23,62). coords(fra1gt_0,250,233,62).  visible(fra1gt_0,62). happensAt(walking(fra1gt_0),62). orientation(fra1gt_7,136,62). coords(fra1gt_7,144,41,62).  visible(fra1gt_7,62). happensAt(inactive(fra1gt_7),62).
orientation(fra1gt_0,23,63). coords(fra1gt_0,251,236,63).  visible(fra1gt_0,63). happensAt(walking(fra1gt_0),63). orientation(fra1gt_7,136,63). coords(fra1gt_7,144,40,63).  visible(fra1gt_7,63). happensAt(inactive(fra1gt_7),63).
orientation(fra1gt_0,34,64). coords(fra1gt_0,251,240,64).  visible(fra1gt_0,64). happensAt(walking(fra1gt_0),64). orientation(fra1gt_7,136,64). coords(fra1gt_7,144,40,64).  visible(fra1gt_7,64). happensAt(inactive(fra1gt_7),64).
orientation(fra1gt_0,39,65). coords(fra1gt_0,251,244,65).  visible(fra1gt_0,65). happensAt(walking(fra1gt_0),65). orientation(fra1gt_7,136,65). coords(fra1gt_7,144,39,65).  visible(fra1gt_7,65). happensAt(inactive(fra1gt_7),65).
orientation(fra1gt_0,39,66). coords(fra1gt_0,254,247,66).  visible(fra1gt_0,66). happensAt(walking(fra1gt_0),66). orientation(fra1gt_7,136,66). coords(fra1gt_7,144,39,66).  visible(fra1gt_7,66). happensAt(inactive(fra1gt_7),66).
orientation(fra1gt_0,39,67). coords(fra1gt_0,258,251,67).  visible(fra1gt_0,67). happensAt(walking(fra1gt_0),67). orientation(fra1gt_7,136,67). coords(fra1gt_7,144,39,67).  visible(fra1gt_7,67). happensAt(inactive(fra1gt_7),67).
orientation(fra1gt_0,39,68). coords(fra1gt_0,261,253,68).  visible(fra1gt_0,68). happensAt(walking(fra1gt_0),68). orientation(fra1gt_7,136,68). coords(fra1gt_7,144,39,68).  visible(fra1gt_7,68). happensAt(inactive(fra1gt_7),68).
orientation(fra1gt_0,39,69). coords(fra1gt_0,264,254,69).  visible(fra1gt_0,69). happensAt(walking(fra1gt_0),69). orientation(fra1gt_7,136,69). coords(fra1gt_7,144,39,69).  visible(fra1gt_7,69). happensAt(inactive(fra1gt_7),69).
orientation(fra1gt_0,39,70). coords(fra1gt_0,266,255,70).  visible(fra1gt_0,70). happensAt(walking(fra1gt_0),70). orientation(fra1gt_7,136,70). coords(fra1gt_7,144,39,70).  visible(fra1gt_7,70). happensAt(inactive(fra1gt_7),70).
orientation(fra1gt_0,39,71). coords(fra1gt_0,267,257,71).  visible(fra1gt_0,71). happensAt(walking(fra1gt_0),71). orientation(fra1gt_7,136,71). coords(fra1gt_7,144,39,71).  visible(fra1gt_7,71). happensAt(inactive(fra1gt_7),71).
orientation(fra1gt_0,39,72). coords(fra1gt_0,269,258,72).  visible(fra1gt_0,72). happensAt(walking(fra1gt_0),72). orientation(fra1gt_7,136,72). coords(fra1gt_7,144,39,72).  visible(fra1gt_7,72). happensAt(inactive(fra1gt_7),72).
orientation(fra1gt_0,46,73). coords(fra1gt_0,270,260,73).  visible(fra1gt_0,73). happensAt(walking(fra1gt_0),73). orientation(fra1gt_7,136,73). coords(fra1gt_7,145,39,73).  visible(fra1gt_7,73). happensAt(inactive(fra1gt_7),73).
orientation(fra1gt_0,46,74). coords(fra1gt_0,271,262,74).  visible(fra1gt_0,74). happensAt(walking(fra1gt_0),74). orientation(fra1gt_7,136,74). coords(fra1gt_7,145,39,74).  visible(fra1gt_7,74). happensAt(inactive(fra1gt_7),74).
orientation(fra1gt_0,46,75). coords(fra1gt_0,272,264,75).  visible(fra1gt_0,75). happensAt(walking(fra1gt_0),75). orientation(fra1gt_7,136,75). coords(fra1gt_7,145,39,75).  visible(fra1gt_7,75). happensAt(inactive(fra1gt_7),75).
orientation(fra1gt_0,46,76). coords(fra1gt_0,273,266,76).  visible(fra1gt_0,76). happensAt(walking(fra1gt_0),76). orientation(fra1gt_7,136,76). coords(fra1gt_7,145,39,76).  visible(fra1gt_7,76). happensAt(inactive(fra1gt_7),76).
orientation(fra1gt_0,46,77). coords(fra1gt_0,274,268,77).  visible(fra1gt_0,77). happensAt(walking(fra1gt_0),77). orientation(fra1gt_7,136,77). coords(fra1gt_7,146,39,77).  visible(fra1gt_7,77). happensAt(inactive(fra1gt_7),77).
orientation(fra1gt_0,57,78). coords(fra1gt_0,274,270,78).  visible(fra1gt_0,78). happensAt(walking(fra1gt_0),78). orientation(fra1gt_7,136,78). coords(fra1gt_7,146,39,78).  visible(fra1gt_7,78). happensAt(inactive(fra1gt_7),78).
orientation(fra1gt_0,57,79). coords(fra1gt_0,275,273,79).  visible(fra1gt_0,79). happensAt(walking(fra1gt_0),79). orientation(fra1gt_7,136,79). coords(fra1gt_7,146,39,79).  visible(fra1gt_7,79). happensAt(walking(fra1gt_7),79).
orientation(fra1gt_0,57,80). coords(fra1gt_0,275,273,80).  visible(fra1gt_0,80). happensAt(walking(fra1gt_0),80). orientation(fra1gt_7,136,80). coords(fra1gt_7,146,39,80).  visible(fra1gt_7,80). happensAt(walking(fra1gt_7),80).
orientation(fra1gt_0,57,81). coords(fra1gt_0,276,273,81).  visible(fra1gt_0,81). happensAt(walking(fra1gt_0),81). orientation(fra1gt_7,136,81). coords(fra1gt_7,146,39,81).  visible(fra1gt_7,81). happensAt(walking(fra1gt_7),81).
orientation(fra1gt_0,57,82). coords(fra1gt_0,276,273,82).  visible(fra1gt_0,82). happensAt(walking(fra1gt_0),82). orientation(fra1gt_7,136,82). coords(fra1gt_7,146,39,82).  visible(fra1gt_7,82). happensAt(walking(fra1gt_7),82).
orientation(fra1gt_0,57,83). coords(fra1gt_0,278,273,83).  visible(fra1gt_0,83). happensAt(walking(fra1gt_0),83). orientation(fra1gt_7,136,83). coords(fra1gt_7,146,39,83).  visible(fra1gt_7,83). happensAt(walking(fra1gt_7),83).
orientation(fra1gt_0,57,84). coords(fra1gt_0,280,274,84).  visible(fra1gt_0,84). happensAt(walking(fra1gt_0),84). orientation(fra1gt_7,136,84). coords(fra1gt_7,146,39,84).  visible(fra1gt_7,84). happensAt(walking(fra1gt_7),84).
#show holdsAt/2.