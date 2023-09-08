
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

	orientation(mws1gt_1,170,49). coords(mws1gt_1,65,193,49).  visible(mws1gt_1,49). happensAt(inactive(mws1gt_1),49). orientation(mws1gt_2,165,49). coords(mws1gt_2,41,147,49).  visible(mws1gt_2,49). happensAt(walking(mws1gt_2),49).
orientation(mws1gt_1,170,50). coords(mws1gt_1,65,193,50).  visible(mws1gt_1,50). happensAt(inactive(mws1gt_1),50). orientation(mws1gt_2,165,50). coords(mws1gt_2,41,147,50).  visible(mws1gt_2,50). happensAt(walking(mws1gt_2),50).
orientation(mws1gt_1,170,51). coords(mws1gt_1,65,193,51).  visible(mws1gt_1,51). happensAt(inactive(mws1gt_1),51). orientation(mws1gt_2,165,51). coords(mws1gt_2,41,147,51).  visible(mws1gt_2,51). happensAt(walking(mws1gt_2),51).
orientation(mws1gt_1,170,52). coords(mws1gt_1,65,193,52).  visible(mws1gt_1,52). happensAt(walking(mws1gt_1),52). orientation(mws1gt_2,165,52). coords(mws1gt_2,41,147,52).  visible(mws1gt_2,52). happensAt(walking(mws1gt_2),52).
orientation(mws1gt_1,170,53). coords(mws1gt_1,65,193,53).  visible(mws1gt_1,53). happensAt(walking(mws1gt_1),53). orientation(mws1gt_2,165,53). coords(mws1gt_2,41,147,53).  visible(mws1gt_2,53). happensAt(walking(mws1gt_2),53).
orientation(mws1gt_1,170,54). coords(mws1gt_1,65,193,54).  visible(mws1gt_1,54). happensAt(walking(mws1gt_1),54). orientation(mws1gt_2,165,54). coords(mws1gt_2,41,147,54).  visible(mws1gt_2,54). happensAt(walking(mws1gt_2),54).
orientation(mws1gt_1,170,55). coords(mws1gt_1,69,193,55).  visible(mws1gt_1,55). happensAt(walking(mws1gt_1),55). orientation(mws1gt_2,165,55). coords(mws1gt_2,41,147,55).  visible(mws1gt_2,55). happensAt(walking(mws1gt_2),55).
orientation(mws1gt_1,172,56). coords(mws1gt_1,69,193,56).  visible(mws1gt_1,56). happensAt(walking(mws1gt_1),56). orientation(mws1gt_2,165,56). coords(mws1gt_2,41,151,56).  visible(mws1gt_2,56). happensAt(walking(mws1gt_2),56).
orientation(mws1gt_1,172,57). coords(mws1gt_1,69,193,57).  visible(mws1gt_1,57). happensAt(walking(mws1gt_1),57). orientation(mws1gt_2,165,57). coords(mws1gt_2,41,154,57).  visible(mws1gt_2,57). happensAt(walking(mws1gt_2),57).
orientation(mws1gt_1,172,58). coords(mws1gt_1,70,193,58).  visible(mws1gt_1,58). happensAt(walking(mws1gt_1),58). orientation(mws1gt_2,165,58). coords(mws1gt_2,41,155,58).  visible(mws1gt_2,58). happensAt(walking(mws1gt_2),58).
orientation(mws1gt_1,172,59). coords(mws1gt_1,70,192,59).  visible(mws1gt_1,59). happensAt(walking(mws1gt_1),59). orientation(mws1gt_2,165,59). coords(mws1gt_2,39,155,59).  visible(mws1gt_2,59). happensAt(walking(mws1gt_2),59).
orientation(mws1gt_1,172,60). coords(mws1gt_1,71,192,60).  visible(mws1gt_1,60). happensAt(walking(mws1gt_1),60). orientation(mws1gt_2,165,60). coords(mws1gt_2,39,157,60).  visible(mws1gt_2,60). happensAt(walking(mws1gt_2),60).
orientation(mws1gt_1,172,61). coords(mws1gt_1,71,192,61).  visible(mws1gt_1,61). happensAt(walking(mws1gt_1),61). orientation(mws1gt_2,165,61). coords(mws1gt_2,39,157,61).  visible(mws1gt_2,61). happensAt(walking(mws1gt_2),61).
orientation(mws1gt_1,172,62). coords(mws1gt_1,71,192,62).  visible(mws1gt_1,62). happensAt(walking(mws1gt_1),62). orientation(mws1gt_2,165,62). coords(mws1gt_2,39,158,62).  visible(mws1gt_2,62). happensAt(walking(mws1gt_2),62).
orientation(mws1gt_1,172,63). coords(mws1gt_1,72,191,63).  visible(mws1gt_1,63). happensAt(walking(mws1gt_1),63). orientation(mws1gt_2,165,63). coords(mws1gt_2,39,158,63).  visible(mws1gt_2,63). happensAt(walking(mws1gt_2),63).
orientation(mws1gt_1,172,64). coords(mws1gt_1,72,191,64).  visible(mws1gt_1,64). happensAt(walking(mws1gt_1),64). orientation(mws1gt_2,165,64). coords(mws1gt_2,38,158,64).  visible(mws1gt_2,64). happensAt(walking(mws1gt_2),64).
orientation(mws1gt_1,172,65). coords(mws1gt_1,72,191,65).  visible(mws1gt_1,65). happensAt(walking(mws1gt_1),65). orientation(mws1gt_2,165,65). coords(mws1gt_2,38,159,65).  visible(mws1gt_2,65). happensAt(walking(mws1gt_2),65).
orientation(mws1gt_1,172,66). coords(mws1gt_1,72,191,66).  visible(mws1gt_1,66). happensAt(walking(mws1gt_1),66). orientation(mws1gt_2,165,66). coords(mws1gt_2,38,159,66).  visible(mws1gt_2,66). happensAt(walking(mws1gt_2),66).
orientation(mws1gt_1,160,67). coords(mws1gt_1,72,190,67).  visible(mws1gt_1,67). happensAt(walking(mws1gt_1),67). orientation(mws1gt_2,165,67). coords(mws1gt_2,37,160,67).  visible(mws1gt_2,67). happensAt(walking(mws1gt_2),67).
orientation(mws1gt_1,160,68). coords(mws1gt_1,72,189,68).  visible(mws1gt_1,68). happensAt(walking(mws1gt_1),68). orientation(mws1gt_2,165,68). coords(mws1gt_2,37,161,68).  visible(mws1gt_2,68). happensAt(walking(mws1gt_2),68).
orientation(mws1gt_1,160,69). coords(mws1gt_1,72,189,69).  visible(mws1gt_1,69). happensAt(walking(mws1gt_1),69). orientation(mws1gt_2,162,69). coords(mws1gt_2,37,162,69).  visible(mws1gt_2,69). happensAt(walking(mws1gt_2),69).
orientation(mws1gt_1,160,70). coords(mws1gt_1,72,189,70).  visible(mws1gt_1,70). happensAt(walking(mws1gt_1),70). orientation(mws1gt_2,162,70). coords(mws1gt_2,37,162,70).  visible(mws1gt_2,70). happensAt(walking(mws1gt_2),70).
orientation(mws1gt_1,160,71). coords(mws1gt_1,72,187,71).  visible(mws1gt_1,71). happensAt(walking(mws1gt_1),71). orientation(mws1gt_2,162,71). coords(mws1gt_2,37,164,71).  visible(mws1gt_2,71). happensAt(walking(mws1gt_2),71).
orientation(mws1gt_1,160,72). coords(mws1gt_1,73,187,72).  visible(mws1gt_1,72). happensAt(walking(mws1gt_1),72). orientation(mws1gt_2,162,72). coords(mws1gt_2,37,166,72).  visible(mws1gt_2,72). happensAt(walking(mws1gt_2),72).
orientation(mws1gt_1,160,73). coords(mws1gt_1,74,187,73).  visible(mws1gt_1,73). happensAt(walking(mws1gt_1),73). orientation(mws1gt_2,162,73). coords(mws1gt_2,37,167,73).  visible(mws1gt_2,73). happensAt(walking(mws1gt_2),73).
#show holdsAt/2.