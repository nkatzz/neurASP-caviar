
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

	orientation(rsfgt_0,150,12). coords(rsfgt_0,129,117,12).  visible(rsfgt_0,12). happensAt(walking(rsfgt_0),12). orientation(rsfgt_1,170,12). coords(rsfgt_1,206,187,12).  visible(rsfgt_1,12). happensAt(walking(rsfgt_1),12).
orientation(rsfgt_0,150,13). coords(rsfgt_0,127,117,13).  visible(rsfgt_0,13). happensAt(walking(rsfgt_0),13). orientation(rsfgt_1,170,13). coords(rsfgt_1,206,187,13).  visible(rsfgt_1,13). happensAt(walking(rsfgt_1),13).
orientation(rsfgt_0,150,14). coords(rsfgt_0,125,117,14).  visible(rsfgt_0,14). happensAt(walking(rsfgt_0),14). orientation(rsfgt_1,170,14). coords(rsfgt_1,206,185,14).  visible(rsfgt_1,14). happensAt(walking(rsfgt_1),14).
orientation(rsfgt_0,150,15). coords(rsfgt_0,122,118,15).  visible(rsfgt_0,15). happensAt(walking(rsfgt_0),15). orientation(rsfgt_1,170,15). coords(rsfgt_1,205,183,15).  visible(rsfgt_1,15). happensAt(walking(rsfgt_1),15).
orientation(rsfgt_0,150,16). coords(rsfgt_0,122,118,16).  visible(rsfgt_0,16). happensAt(walking(rsfgt_0),16). orientation(rsfgt_1,170,16). coords(rsfgt_1,204,183,16).  visible(rsfgt_1,16). happensAt(walking(rsfgt_1),16).
orientation(rsfgt_0,150,17). coords(rsfgt_0,121,119,17).  visible(rsfgt_0,17). happensAt(walking(rsfgt_0),17). orientation(rsfgt_1,170,17). coords(rsfgt_1,204,181,17).  visible(rsfgt_1,17). happensAt(walking(rsfgt_1),17).
orientation(rsfgt_0,150,18). coords(rsfgt_0,120,119,18).  visible(rsfgt_0,18). happensAt(walking(rsfgt_0),18). orientation(rsfgt_1,170,18). coords(rsfgt_1,203,180,18).  visible(rsfgt_1,18). happensAt(walking(rsfgt_1),18).
orientation(rsfgt_0,150,19). coords(rsfgt_0,120,122,19).  visible(rsfgt_0,19). happensAt(walking(rsfgt_0),19). orientation(rsfgt_1,170,19). coords(rsfgt_1,202,179,19).  visible(rsfgt_1,19). happensAt(walking(rsfgt_1),19).
orientation(rsfgt_0,150,20). coords(rsfgt_0,119,122,20).  visible(rsfgt_0,20). happensAt(walking(rsfgt_0),20). orientation(rsfgt_1,170,20). coords(rsfgt_1,201,178,20).  visible(rsfgt_1,20). happensAt(walking(rsfgt_1),20).
orientation(rsfgt_0,150,21). coords(rsfgt_0,118,122,21).  visible(rsfgt_0,21). happensAt(walking(rsfgt_0),21). orientation(rsfgt_1,170,21). coords(rsfgt_1,198,175,21).  visible(rsfgt_1,21). happensAt(walking(rsfgt_1),21).
orientation(rsfgt_0,150,22). coords(rsfgt_0,118,122,22).  visible(rsfgt_0,22). happensAt(walking(rsfgt_0),22). orientation(rsfgt_1,170,22). coords(rsfgt_1,197,174,22).  visible(rsfgt_1,22). happensAt(walking(rsfgt_1),22).
orientation(rsfgt_0,150,23). coords(rsfgt_0,117,124,23).  visible(rsfgt_0,23). happensAt(walking(rsfgt_0),23). orientation(rsfgt_1,170,23). coords(rsfgt_1,196,172,23).  visible(rsfgt_1,23). happensAt(walking(rsfgt_1),23).
orientation(rsfgt_0,150,24). coords(rsfgt_0,115,124,24).  visible(rsfgt_0,24). happensAt(walking(rsfgt_0),24). orientation(rsfgt_1,170,24). coords(rsfgt_1,192,167,24).  visible(rsfgt_1,24). happensAt(walking(rsfgt_1),24).
orientation(rsfgt_0,150,25). coords(rsfgt_0,113,124,25).  visible(rsfgt_0,25). happensAt(walking(rsfgt_0),25). orientation(rsfgt_1,170,25). coords(rsfgt_1,191,166,25).  visible(rsfgt_1,25). happensAt(walking(rsfgt_1),25).
orientation(rsfgt_0,150,26). coords(rsfgt_0,112,124,26).  visible(rsfgt_0,26). happensAt(walking(rsfgt_0),26). orientation(rsfgt_1,170,26). coords(rsfgt_1,191,165,26).  visible(rsfgt_1,26). happensAt(walking(rsfgt_1),26).
orientation(rsfgt_0,150,27). coords(rsfgt_0,109,124,27).  visible(rsfgt_0,27). happensAt(walking(rsfgt_0),27). orientation(rsfgt_1,170,27). coords(rsfgt_1,189,164,27).  visible(rsfgt_1,27). happensAt(walking(rsfgt_1),27).
orientation(rsfgt_0,150,28). coords(rsfgt_0,108,124,28).  visible(rsfgt_0,28). happensAt(walking(rsfgt_0),28). orientation(rsfgt_1,170,28). coords(rsfgt_1,189,163,28).  visible(rsfgt_1,28). happensAt(walking(rsfgt_1),28).
orientation(rsfgt_0,150,29). coords(rsfgt_0,107,124,29).  visible(rsfgt_0,29). happensAt(walking(rsfgt_0),29). orientation(rsfgt_1,170,29). coords(rsfgt_1,189,162,29).  visible(rsfgt_1,29). happensAt(walking(rsfgt_1),29).
orientation(rsfgt_0,150,30). coords(rsfgt_0,106,125,30).  visible(rsfgt_0,30). happensAt(walking(rsfgt_0),30). orientation(rsfgt_1,170,30). coords(rsfgt_1,189,161,30).  visible(rsfgt_1,30). happensAt(walking(rsfgt_1),30).
orientation(rsfgt_0,150,31). coords(rsfgt_0,106,125,31).  visible(rsfgt_0,31). happensAt(walking(rsfgt_0),31). orientation(rsfgt_1,150,31). coords(rsfgt_1,189,160,31).  visible(rsfgt_1,31). happensAt(walking(rsfgt_1),31).
orientation(rsfgt_0,150,32). coords(rsfgt_0,105,125,32).  visible(rsfgt_0,32). happensAt(walking(rsfgt_0),32). orientation(rsfgt_1,150,32). coords(rsfgt_1,189,159,32).  visible(rsfgt_1,32). happensAt(walking(rsfgt_1),32).
orientation(rsfgt_0,150,33). coords(rsfgt_0,104,125,33).  visible(rsfgt_0,33). happensAt(walking(rsfgt_0),33). orientation(rsfgt_1,150,33). coords(rsfgt_1,188,158,33).  visible(rsfgt_1,33). happensAt(walking(rsfgt_1),33).
orientation(rsfgt_0,150,34). coords(rsfgt_0,104,127,34).  visible(rsfgt_0,34). happensAt(walking(rsfgt_0),34). orientation(rsfgt_1,150,34). coords(rsfgt_1,188,157,34).  visible(rsfgt_1,34). happensAt(walking(rsfgt_1),34).
orientation(rsfgt_0,150,35). coords(rsfgt_0,104,127,35).  visible(rsfgt_0,35). happensAt(walking(rsfgt_0),35). orientation(rsfgt_1,150,35). coords(rsfgt_1,188,155,35).  visible(rsfgt_1,35). happensAt(walking(rsfgt_1),35).
orientation(rsfgt_0,160,36). coords(rsfgt_0,103,127,36).  visible(rsfgt_0,36). happensAt(walking(rsfgt_0),36). orientation(rsfgt_1,150,36). coords(rsfgt_1,185,152,36).  visible(rsfgt_1,36). happensAt(walking(rsfgt_1),36).
#show holdsAt/2.