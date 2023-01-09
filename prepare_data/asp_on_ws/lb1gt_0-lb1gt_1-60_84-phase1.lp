
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

	orientation(lb1gt_0,148,60). coords(lb1gt_0,181,119,60).  visible(lb1gt_0,60). happensAt(active(lb1gt_0),60). orientation(lb1gt_1,147,60). coords(lb1gt_1,74,101,60).  visible(lb1gt_1,60). happensAt(walking(lb1gt_1),60).
orientation(lb1gt_0,148,61). coords(lb1gt_0,180,119,61).  visible(lb1gt_0,61). happensAt(active(lb1gt_0),61). orientation(lb1gt_1,147,61). coords(lb1gt_1,74,101,61).  visible(lb1gt_1,61). happensAt(walking(lb1gt_1),61).
orientation(lb1gt_0,148,62). coords(lb1gt_0,180,119,62).  visible(lb1gt_0,62). happensAt(active(lb1gt_0),62). orientation(lb1gt_1,147,62). coords(lb1gt_1,74,102,62).  visible(lb1gt_1,62). happensAt(walking(lb1gt_1),62).
orientation(lb1gt_0,148,63). coords(lb1gt_0,179,119,63).  visible(lb1gt_0,63). happensAt(active(lb1gt_0),63). orientation(lb1gt_1,147,63). coords(lb1gt_1,74,102,63).  visible(lb1gt_1,63). happensAt(walking(lb1gt_1),63).
orientation(lb1gt_0,148,64). coords(lb1gt_0,178,119,64).  visible(lb1gt_0,64). happensAt(active(lb1gt_0),64). orientation(lb1gt_1,147,64). coords(lb1gt_1,74,102,64).  visible(lb1gt_1,64). happensAt(walking(lb1gt_1),64).
orientation(lb1gt_0,148,65). coords(lb1gt_0,177,118,65).  visible(lb1gt_0,65). happensAt(active(lb1gt_0),65). orientation(lb1gt_1,147,65). coords(lb1gt_1,74,102,65).  visible(lb1gt_1,65). happensAt(walking(lb1gt_1),65).
orientation(lb1gt_0,148,66). coords(lb1gt_0,176,118,66).  visible(lb1gt_0,66). happensAt(active(lb1gt_0),66). orientation(lb1gt_1,147,66). coords(lb1gt_1,74,102,66).  visible(lb1gt_1,66). happensAt(walking(lb1gt_1),66).
orientation(lb1gt_0,148,67). coords(lb1gt_0,176,117,67).  visible(lb1gt_0,67). happensAt(active(lb1gt_0),67). orientation(lb1gt_1,147,67). coords(lb1gt_1,74,103,67).  visible(lb1gt_1,67). happensAt(walking(lb1gt_1),67).
orientation(lb1gt_0,148,68). coords(lb1gt_0,176,118,68).  visible(lb1gt_0,68). happensAt(active(lb1gt_0),68). orientation(lb1gt_1,147,68). coords(lb1gt_1,74,103,68).  visible(lb1gt_1,68). happensAt(walking(lb1gt_1),68).
orientation(lb1gt_0,148,69). coords(lb1gt_0,176,117,69).  visible(lb1gt_0,69). happensAt(active(lb1gt_0),69). orientation(lb1gt_1,147,69). coords(lb1gt_1,73,103,69).  visible(lb1gt_1,69). happensAt(walking(lb1gt_1),69).
orientation(lb1gt_0,148,70). coords(lb1gt_0,175,116,70).  visible(lb1gt_0,70). happensAt(active(lb1gt_0),70). orientation(lb1gt_1,147,70). coords(lb1gt_1,73,103,70).  visible(lb1gt_1,70). happensAt(walking(lb1gt_1),70).
orientation(lb1gt_0,148,71). coords(lb1gt_0,175,115,71).  visible(lb1gt_0,71). happensAt(active(lb1gt_0),71). orientation(lb1gt_1,147,71). coords(lb1gt_1,74,104,71).  visible(lb1gt_1,71). happensAt(walking(lb1gt_1),71).
orientation(lb1gt_0,148,72). coords(lb1gt_0,175,113,72).  visible(lb1gt_0,72). happensAt(active(lb1gt_0),72). orientation(lb1gt_1,147,72). coords(lb1gt_1,74,104,72).  visible(lb1gt_1,72). happensAt(walking(lb1gt_1),72).
orientation(lb1gt_0,148,73). coords(lb1gt_0,173,112,73).  visible(lb1gt_0,73). happensAt(active(lb1gt_0),73). orientation(lb1gt_1,147,73). coords(lb1gt_1,75,104,73).  visible(lb1gt_1,73). happensAt(walking(lb1gt_1),73).
orientation(lb1gt_0,148,74). coords(lb1gt_0,172,112,74).  visible(lb1gt_0,74). happensAt(active(lb1gt_0),74). orientation(lb1gt_1,147,74). coords(lb1gt_1,75,105,74).  visible(lb1gt_1,74). happensAt(walking(lb1gt_1),74).
orientation(lb1gt_0,145,75). coords(lb1gt_0,170,111,75).  visible(lb1gt_0,75). happensAt(active(lb1gt_0),75). orientation(lb1gt_1,147,75). coords(lb1gt_1,75,105,75).  visible(lb1gt_1,75). happensAt(walking(lb1gt_1),75).
orientation(lb1gt_0,145,76). coords(lb1gt_0,169,110,76).  visible(lb1gt_0,76). happensAt(active(lb1gt_0),76). orientation(lb1gt_1,147,76). coords(lb1gt_1,75,106,76).  visible(lb1gt_1,76). happensAt(walking(lb1gt_1),76).
orientation(lb1gt_0,145,77). coords(lb1gt_0,168,110,77).  visible(lb1gt_0,77). happensAt(active(lb1gt_0),77). orientation(lb1gt_1,147,77). coords(lb1gt_1,75,107,77).  visible(lb1gt_1,77). happensAt(walking(lb1gt_1),77).
orientation(lb1gt_0,145,78). coords(lb1gt_0,167,110,78).  visible(lb1gt_0,78). happensAt(active(lb1gt_0),78). orientation(lb1gt_1,147,78). coords(lb1gt_1,75,107,78).  visible(lb1gt_1,78). happensAt(walking(lb1gt_1),78).
orientation(lb1gt_0,145,79). coords(lb1gt_0,167,109,79).  visible(lb1gt_0,79). happensAt(active(lb1gt_0),79). orientation(lb1gt_1,147,79). coords(lb1gt_1,75,108,79).  visible(lb1gt_1,79). happensAt(walking(lb1gt_1),79).
orientation(lb1gt_0,145,80). coords(lb1gt_0,167,109,80).  visible(lb1gt_0,80). happensAt(active(lb1gt_0),80). orientation(lb1gt_1,147,80). coords(lb1gt_1,75,109,80).  visible(lb1gt_1,80). happensAt(walking(lb1gt_1),80).
orientation(lb1gt_0,145,81). coords(lb1gt_0,167,109,81).  visible(lb1gt_0,81). happensAt(active(lb1gt_0),81). orientation(lb1gt_1,147,81). coords(lb1gt_1,75,109,81).  visible(lb1gt_1,81). happensAt(walking(lb1gt_1),81).
orientation(lb1gt_0,145,82). coords(lb1gt_0,166,109,82).  visible(lb1gt_0,82). happensAt(active(lb1gt_0),82). orientation(lb1gt_1,147,82). coords(lb1gt_1,75,109,82).  visible(lb1gt_1,82). happensAt(walking(lb1gt_1),82).
orientation(lb1gt_0,145,83). coords(lb1gt_0,166,108,83).  visible(lb1gt_0,83). happensAt(active(lb1gt_0),83). orientation(lb1gt_1,147,83). coords(lb1gt_1,75,109,83).  visible(lb1gt_1,83). happensAt(walking(lb1gt_1),83).
orientation(lb1gt_0,145,84). coords(lb1gt_0,166,108,84).  visible(lb1gt_0,84). happensAt(active(lb1gt_0),84). orientation(lb1gt_1,147,84). coords(lb1gt_1,75,109,84).  visible(lb1gt_1,84). happensAt(walking(lb1gt_1),84).
#show holdsAt/2.